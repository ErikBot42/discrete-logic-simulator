use std::time::Instant;
use wgpu::util::DeviceExt;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

// traces = 2000*2000
// index data = 32 * traces
// trace data = traces
// color data = traces * 32 * 2
// state data = 1 * gates ~ 0
//
// Total if enum: 20 000 000 = 19 MiB
// Total if col:  48 000 000 = 46 MiB
//
// Max gates: 300 000 = 37500 B = 37 KB of dynamic data per frame

//#[repr(C)]
//#[derive(Copy, Clone, Debug)]
//struct Vertex {
//    position: [f32; 3],
//    color: [f32; 3],
//}

//struct SimParams {
//    delta: f32,
//    foo: f32,
//}

// IN:
// matrix of gate ids.
// matrix of trace enum vals.
//
// SIM:
// bitvec of state.

pub struct TraceInfo {
    pub color: [u8; 4],
    // color_on: [u8; 4]
    // color_off: [u8; 4]
    // pub id: u8,
}

pub struct RenderInput {
    pub trace_info: Vec<TraceInfo>,
    pub traces: Vec<u8>,
    pub gate_ids: Vec<u32>,
    pub width: usize,
    pub height: usize,
}
impl RenderInput {
    fn validate_ranges(&self) {
        assert_eq!(self.width * self.height, self.traces.len());
        assert_eq!(self.width * self.height, self.gate_ids.len());
    }
}

struct SimParams {
    max_x: f32,
    max_y: f32,
    offset_x: f32,
    offset_y: f32,
    zoom_x: f32,
    zoom_y: f32,
}
impl SimParams {
    fn as_arr(&self) -> [f32; 6] {
        [
            self.max_x,
            self.max_y,
            self.offset_x,
            self.offset_y,
            self.zoom_x,
            self.zoom_y,
        ]
    }
}
#[derive(Default)]
struct KeyStates {
    /// up (+), down (-)
    u: f32,
    /// right (+), left (-)
    r: f32,
    /// out (+), in (-)
    z: f32,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    window_size: winit::dpi::PhysicalSize<u32>,
    window: Window,

    render_pipeline: wgpu::RenderPipeline,

    y: Vec<u32>,
    y_buffer: wgpu::Buffer,

    trace_buffer: wgpu::Buffer,   // 32-bit starting out ( -> 8 bit )
    gate_id_buffer: wgpu::Buffer, // 32-bit starting out ( -> 24 bit )
    trace_color_buffer: wgpu::Buffer,

    sim_params: SimParams,
    sim_param_buffer: wgpu::Buffer,

    bind_group: wgpu::BindGroup,

    keystates: KeyStates,

    last_update: Instant,
}

impl State {
    fn update_zoom(&mut self, delta: f32) {
        self.sim_params.zoom_y += delta;
    }
    // Creating some of the wgpu types requires async code
    async fn new(window: Window, render_input: RenderInput) -> Self {
        render_input.validate_ranges();
        let sim_params = SimParams {
            max_x: render_input.width as f32,
            max_y: render_input.height as f32,
            offset_x: 0.0,
            offset_y: 0.0,
            zoom_x: 1.0,
            zoom_y: render_input.height as f32,
        };

        let window_size = window.inner_size();
        // The instance is a handle to our GPU
        let instance = wgpu::Instance::default();
        // Safety: lifetime >= window
        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let (adapter, device, queue) = prep_device(instance, &surface).await;
        let (_surface_capabilites, surface_config) =
            prep_surface(&surface, adapter, window_size, &device);

        let shader: wgpu::ShaderModule =
            device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let trace_colors: Vec<_> = render_input
            .trace_info
            .iter()
            .map(|t| u32::from_le_bytes(t.color))
            .collect();
        let trace_color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&trace_colors),
            usage: wgpu::BufferUsages::STORAGE // TODO: reduce this
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let traces: Vec<u32> = render_input.traces.iter().map(|&i| i.into()).collect();
        let trace_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&traces),
            usage: wgpu::BufferUsages::STORAGE // TODO: reduce this
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let gate_ids = render_input.gate_ids;
        let gate_id_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&gate_ids),
            usage: wgpu::BufferUsages::STORAGE // TODO: reduce this
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let y: Vec<u32> = (0..(1024 * 256)).into_iter().map(|x| x).collect();
        let y_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&y),
            usage: wgpu::BufferUsages::STORAGE // TODO: reduce this
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let sim_param_data = sim_params.as_arr().to_vec();
        let sim_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim param buffer"),
            contents: bytemuck::cast_slice(&sim_param_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()), // TODO: what
                    },
                    count: None, // TODO: what
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            u64::try_from(sim_param_data.len() * std::mem::size_of::<f32>())
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()), // TODO: what
                    },
                    count: None, // TODO: what
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()), // TODO: what
                    },
                    count: None, // TODO: what
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()), // TODO: what
                    },
                    count: None, // TODO: what
                },
            ],
        });
        /*wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &y_buffer,
            offset: 0,
            size: None, // use rest of buffer
                        // Some(std::num::NonZeroU64::new(64).unwrap()), // TODO: what
        })*/
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sim_param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: trace_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gate_id_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: trace_color_buffer.as_entire_binding(),
                },
            ],
            label: Some("bind group"),
            layout: &bind_group_layout,
        });

        let render_pipeline_layout: wgpu::PipelineLayout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline =
            create_render_pipeline(&device, render_pipeline_layout, shader, &surface_config);

        Self {
            surface,
            device,
            queue,
            surface_config,
            window_size,
            window,
            render_pipeline,
            y,
            y_buffer,
            bind_group,
            sim_params,
            sim_param_buffer,
            keystates: KeyStates::default(),
            last_update: Instant::now(),
            trace_buffer,
            gate_id_buffer,
            trace_color_buffer,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.window_size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        if let WindowEvent::KeyboardInput {
            device_id: _,
            input:
                KeyboardInput {
                    //scancode,
                    state,
                    virtual_keycode: Some(keycode),
                    ..
                },
            is_synthetic: _,
        } = event
        {
            let k = &mut self.keystates;
            let (p, n) = match state {
                ElementState::Pressed => (1.0, -1.0),
                ElementState::Released => (0.0, 0.0),
            };
            use VirtualKeyCode::*;
            match keycode {
                W | F => k.u = p,
                A | R => k.r = n,
                S => k.u = n,
                D | T => k.r = p,
                J => k.z = n,
                K => k.z = p,
                _ => (),
            }
        };
        false
    }

    fn update(&mut self) {
        for y in self.y.iter_mut() {
            *y += 1;
        }

        let wsize = self.window.inner_size();
        let ratio = wsize.width as f32 / wsize.height as f32;
        self.sim_params.zoom_x = self.sim_params.zoom_y * ratio;

        let scale = self.sim_params.zoom_y;
        let dt = self.last_update.elapsed().as_secs_f32();
        let ds = scale * dt;

        self.last_update = Instant::now();
        self.sim_params.offset_x -= self.keystates.r * ds;
        self.sim_params.offset_y -= self.keystates.u * ds;

        let dz = self.sim_params.zoom_y * self.keystates.z * dt;
        self.sim_params.zoom_y += dz;
        self.sim_params.offset_x += ratio * dz / 2.0;
        self.sim_params.offset_y += dz / 2.0;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.queue.write_buffer(
            &self.y_buffer,
            0, /* offset */
            bytemuck::cast_slice(&self.y),
        );
        let sim_params_arr = self.sim_params.as_arr();
        self.queue.write_buffer(
            &self.sim_param_buffer,
            0, /* offset */
            bytemuck::cast_slice(&sim_params_arr),
        );

        //let y: Vec<u32> = (0..1000).into_iter().map(|x| x).collect();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_bind_group(0, &self.bind_group, &[]);

            render_pass.set_pipeline(&self.render_pipeline); // 2.
            render_pass.draw(0..3, 0..1); // 3.
        }
        // retive from shader
        //encoder.copy_buffer_to_buffer(
        //    &self.y_buffer,
        //    0,
        //    &self.staging_buffer,
        //    0,
        //    slice_size(&self.y) as wgpu::BufferAddress,
        //);

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn prep_surface(
    surface: &wgpu::Surface,
    adapter: wgpu::Adapter,
    size: winit::dpi::PhysicalSize<u32>,
    device: &wgpu::Device,
) -> (wgpu::SurfaceCapabilities, wgpu::SurfaceConfiguration) {
    let surface_capabilites = surface.get_capabilities(&adapter);
    // Shader code here assumes an sRGB surface texture. Using a different
    // one will result all the colors coming out darker. If you want to support non
    // sRGB surfaces, you'll need to account for that when drawing to the frame.
    let surface_format = surface_capabilites
        .formats
        .iter()
        .copied()
        .find(|f| f.describe().srgb)
        .unwrap_or(surface_capabilites.formats[0]);

    let surface_config: wgpu::SurfaceConfiguration = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: surface_capabilites.present_modes[0],
        alpha_mode: surface_capabilites.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(device, &surface_config);
    (surface_capabilites, surface_config)
}

async fn prep_device(
    instance: wgpu::Instance,
    surface: &wgpu::Surface,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
            },
            None,
        )
        .await
        .unwrap();
    (adapter, device, queue)
}

fn create_render_pipeline(
    device: &wgpu::Device,
    render_pipeline_layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    surface_config: &wgpu::SurfaceConfiguration,
) -> wgpu::RenderPipeline {
    let render_pipeline: wgpu::RenderPipeline =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
    render_pipeline
}

pub async fn run(render_input: RenderInput) {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = State::new(window, render_input).await;
    let mut last_frame_inst = Instant::now();
    let (mut frame_count, mut accum_time) = (0, 0.0);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                state.update();

                accum_time += last_frame_inst.elapsed().as_secs_f32();
                last_frame_inst = Instant::now();
                frame_count += 1;
                if frame_count == 200 {
                    let avg_frame_time = accum_time * 1000.0 / frame_count as f32;
                    println!("Avg frame time {}ms", avg_frame_time);
                    accum_time = 0.0;
                    frame_count = 0;
                }

                match state.render() {
                    Ok(_) => {},
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.window_size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            },
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                state.window().request_redraw();
            },
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                if !state.input(event) {
                    // UPDATED!
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        },
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        },
                        _ => {},
                    }
                }
            },
            _ => {},
        }
    });
}

fn slice_size<T>(slice: &[T]) -> usize {
    std::mem::size_of::<T>() * slice.len()
}
