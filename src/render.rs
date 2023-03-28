use std::mem;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

// split buffers: 520.1964

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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 2],   // TODO: omit third coordinate
    tex_coords: [f32; 2], // 0..1
}
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}
impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2];
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
    fn get_vertices(max_width: usize, max_height: usize) -> [Vertex; 6] {
        let f: f32 = 1.0;
        let positions = [[f, f], [-f, -f], [f, -f], [-f, f], [-f, -f], [f, f]];
        positions.map(|position| Vertex {
            position,
            tex_coords: [
                (position[0] + f) / (2.0 * f) * (max_width as f32),
                (-position[1] + f) / (2.0 * f) * (max_height as f32),
            ],
        })
    }
}

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
use crate::logic::RenderSim;

// let sim run,
// interrupt,
// swap buffers,
// resume sim

// fn foo() {
//     fn run() {}
//     let handle = run();
//
//     handle.pause();
//
//     handle.get_state();
//
//     handle.step();
//
//     handle.resume();
//
//     // later:
//     let state = handle.get_state();
// }

use std::marker::Send;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

#[derive(Debug)]
enum RenderSimTask {
    Exit,
    CopyData,
    Run,
    Step,
    Pause,
}

// things left to evaluate:
// # async mpsc variants.
// Will this allways allocate?
// How will load balancing work?
// Guaranteed to only send valid state.
// reciv: try_recv.unwrap() <- detect when stuff is wrong.
//
// # buffer pair:
// shared atomic.
// if incremented, sim thread writes to a buffer and swaps what buffer it will write to next
// this means that there is 1 valid buffer avaliable to read from.
// sim thread activates condvar when write ends.
// next iteration, render thread waits for the condvar to know when to start reading.
//
// when reading, render thread swaps vec objects and then issues a sync operation.
//
// UNRESOLVED: sim optimizations based on all vecs having the same length.
use std::sync::mpsc;
struct RenderSimController {
    interrupt: Arc<AtomicUsize>,
    recv: mpsc::Receiver<(usize, Vec<u64>)>,
    tick_counter: usize,
    last_counter_reset: Instant,
    generation: usize,
}
impl RenderSimController {
    fn start_sim_thread<S: RenderSim + Send + 'static>(
        sim: S,
        interrupt: Arc<AtomicUsize>,
        send: mpsc::Sender<(usize, Vec<u64>)>,
    ) {
        std::thread::spawn(move || {
            let mut sim = sim;
            let interrupt = interrupt;
            let mut generation = 0;
            loop {
                let mut tick_counter = 0;
                let max_ticks = 1;
                while interrupt.load(Ordering::Acquire) == generation {
                    if tick_counter < max_ticks {
                        sim.rupdate();
                        tick_counter += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
                generation = generation.wrapping_add(1);
                let mut v = Vec::new(); // allocation :(
                sim.get_state_in(&mut v);
                send.send((tick_counter, v)).unwrap();
            }
        });
    }
    fn new<S: RenderSim + Send + 'static>(sim: S) -> Self {
        let (send, recv) = mpsc::channel();

        let generation = 5; // make sure there is some buffering.

        let interrupt = Arc::new(AtomicUsize::new(generation));

        Self::start_sim_thread(sim, interrupt.clone(), send);
        Self {
            interrupt,
            recv,
            tick_counter: 0,
            last_counter_reset: Instant::now(),
            generation,
        }
    }
    fn get_state_in(&mut self, state: &mut Vec<u64>) {
        self.generation += 1;
        self.interrupt.store(self.generation, Ordering::Release);

        // will just panic instead of block.
        match self.recv.try_recv() {
            Ok((count, mut new_state)) => {
                mem::swap(state, &mut new_state);
                self.tick_counter += count;
            },
            Err(mpsc::TryRecvError::Empty) => {
                // this will implicitly increase the size of the queue because generation increases
            },
            Err(mpsc::TryRecvError::Disconnected) => panic!("disconnected"),
        }
    }
    fn get_counter(&mut self) -> (usize, Duration) {
        (
            mem::replace(&mut self.tick_counter, 0),
            mem::replace(&mut self.last_counter_reset, Instant::now()).elapsed(),
        )
    }
    // stop sim from calling update
    fn pause(&mut self) {
        todo!()
    }
    // resume to max speed
    fn resume(&mut self) {
        todo!()
    }
    // only on paused sim
    fn step(&mut self, steps: usize) {
        todo!()
    }
}
/*struct RenderSimController {
    barrier: Arc<Barrier>,
    interrupt: Arc<AtomicBool>,
    shared_data: Arc<Mutex<(RenderSimTask, Vec<u64>, usize)>>,
    tick_counter: usize,
    last_counter_reset: Instant,
}
impl RenderSimController {
    /// start paused
    fn new<S: RenderSim + Send + 'static>(sim: S) -> Self {
        let barrier = Arc::new(Barrier::new(2));
        let interrupt = Arc::new(AtomicBool::new(true));
        let shared_data = Arc::new(Mutex::new((RenderSimTask::Pause, Vec::<u64>::new(), 0)));

        let interrupt_clone = interrupt.clone();
        let shared_data_clone = shared_data.clone();
        let barrier_clone = barrier.clone();
        std::thread::spawn(move || {
            let mut sim = sim;
            let interrupt = interrupt_clone;
            let shared_data = shared_data_clone;
            let barrier = barrier_clone;
            let mut ticks_since_copy = 0;
            loop {
                while interrupt.load(Ordering::Acquire) {
                    sim.rupdate();
                    ticks_since_copy += 1;
                }
                //println!("SIM: interrupted");
                interrupt.store(true, Ordering::Release);
                let mut data = shared_data.lock().unwrap();
                data.2 += std::mem::replace(&mut ticks_since_copy, 0);
                sim.get_state_in(&mut data.1);
                barrier.wait();

                //println!("SIM: get lock");
                //let mut data = shared_data.lock().unwrap();
                ////println!("SIM task: {:?}", data.0);
                //match data.0 {
                //    RenderSimTask::Exit => break,
                //    RenderSimTask::CopyData => {
                //        sim.get_state_in(&mut data.1);
                //        // singnal data copy done.
                //        data.0 = RenderSimTask::Pause;
                //        data.2 += ticks_since_copy;
                //        ticks_since_copy = 0;
                //        drop(data);
                //        barrier.wait();
                //    },
                //    RenderSimTask::Run => {
                //        drop(data);
                //        //println!("SIM: start sim");
                //        sim.rupdate();
                //        ticks_since_copy += 1;
                //        while interrupt.load(Ordering::Acquire) {}
                //        //println!("SIM: interrupted");
                //        interrupt.store(true, Ordering::Release);
                //        // set next task beforehand
                //    },
                //    RenderSimTask::Pause => {
                //        drop(data);
                //        // wait to resume, set next before resume
                //        //println!("SIM: paused, waiting");
                //        barrier.wait();
                //    },
                //    RenderSimTask::Step => {
                //        data.0 = RenderSimTask::Pause;
                //        sim.rupdate();
                //        drop(data);
                //    },
                //}
            }
        });
        Self {
            barrier,
            interrupt,
            shared_data,
            tick_counter: 0,
            last_counter_reset: Instant::now(),
        }
    }
    /// pause, get copy of state, resume
    fn get_state_in(&mut self, state: &mut Vec<u64>) {
        self.interrupt.store(false, Ordering::Release);
        self.barrier.wait();
        let mut lock = self.shared_data.lock().unwrap();
        self.tick_counter += lock.2;
        lock.2 = 0;
        std::mem::swap(state, &mut lock.1);

        //self.shared_data.lock().unwrap().0 = RenderSimTask::CopyData;
        //self.interrupt.store(false, Ordering::Release);
        //self.barrier.wait();
        //let mut lock = self.shared_data.lock().unwrap();
        //std::mem::swap(state, &mut lock.1);
        //lock.0 = RenderSimTask::Run;
        //self.tick_counter += lock.2;
        //lock.2 = 0;
    }
    fn get_counter(&mut self) -> (usize, Duration) {
        (
            std::mem::replace(&mut self.tick_counter, 0),
            std::mem::replace(&mut self.last_counter_reset, Instant::now()).elapsed(),
        )
    }

    // resume sim, unset condvar
    //fn resume(&mut self) {
    //    todo!()
    //}

    ///// pause sim, set condvar
    //fn pause(&mut self) {
    //    todo!()
    //}
    // run 1 tick = pause, send thing, resume
    // fn step(&mut self) {}
}*/

pub struct TraceInfo {
    pub color: [u8; 4],
    pub color_on: [u8; 4],
    pub color_off: [u8; 4],
    // pub id: u8,
}

pub struct RenderInput<S: RenderSim> {
    pub trace_info: Vec<TraceInfo>,
    pub traces: Vec<u8>,
    pub gate_ids: Vec<u32>,
    pub width: usize,
    pub height: usize,
    pub sim: S,
}
impl<S: RenderSim> RenderInput<S> {
    fn validate_ranges(&self) {
        assert_eq!(self.width * self.height, self.traces.len());
        assert_eq!(self.width * self.height, self.gate_ids.len());
    }
}

/// 32-bit id -> 24-bit id + 8-bit trace
fn pack_single(gate_id: u32, trace: u8) -> u32 {
    let id_mask = u32::MAX >> 8;

    let masked_id = id_mask & gate_id;
    assert_eq!(masked_id, gate_id);

    gate_id | (u32::from(trace) << 24)
}
#[derive(Default, Copy, Clone, Debug)]
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
#[derive(Debug)]
struct ViewState {
    // trace is center of screen.
    // zoom, trace in trace scale.
    // zoom = "height" in trace scale
    max_x: f32,
    max_y: f32,

    trace_x: f32,
    trace_y: f32,
    zoom: f32,
}
impl ViewState {
    fn new(width: usize, height: usize) -> Self {
        let width = width as f32;
        let height = height as f32;
        Self {
            max_x: width,
            max_y: height,
            trace_x: width / 2.0,
            trace_y: height / 2.0,
            zoom: height,
        }
    }
    fn as_sim_params(&self, ratio: f32) -> SimParams {
        // vert = position * zoom + offset
        // (vert - offset) = position * zoom
        // (vert - offset) / zoom = position * zoom
        // vert / zoom - (offset / zoom) = position
        // vert * (1 / zoom) - (offset / zoom) = position
        SimParams {
            max_x: self.max_x,
            max_y: self.max_y,
            offset_x: -self.trace_x / self.max_x * 2.0 + 1.0,
            offset_y: -self.trace_y / self.max_y * 2.0 + 1.0,
            zoom_x: self.max_x / (self.zoom * ratio),
            //zoom_x: self.max_y / (self.zoom * ratio * (self.max_x / self.max_y)),
            zoom_y: self.max_y / self.zoom,
        }
    }
}
#[derive(Default)]
struct KeyStates {
    /// up (+), down (-)
    up: f32,
    /// right (+), left (-)
    right: f32,
    /// out (+), in (-)
    zoom: f32,
    /// reverse (time)
    reverse: bool,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    window_size: winit::dpi::PhysicalSize<u32>,
    window: Window,

    render_pipeline: wgpu::RenderPipeline,

    bit_state: Vec<u64>,
    bit_state_buffer: wgpu::Buffer,

    //trace_buffer: wgpu::Buffer,   // 32-bit starting out ( -> 8 bit )
    //gate_id_buffer: wgpu::Buffer, // 32-bit starting out ( -> 24 bit )
    //trace_color_buffer: wgpu::Buffer,
    sim_params: SimParams,
    sim_param_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    keystates: KeyStates,
    last_update: Instant,
    sim_controller: RenderSimController,

    last_print: Instant,

    state_history: Vec<Vec<u64>>,

    vertex_buffer: wgpu::Buffer,

    num_vertices: u32,

    view: ViewState,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new<S: RenderSim + Send + 'static>(
        window: Window,
        render_input: RenderInput<S>,
    ) -> Self {
        render_input.validate_ranges();
        let view = ViewState::new(render_input.width, render_input.height);
        //ViewState {
        //    trace_x: render_input.width as f32 / 2.0,
        //    trace_y: render_input.height as f32 / 2.0,
        //    zoom: render_input.height as f32,
        //    max_x: render_input.width as f32,
        //    max_y: render_input.height as f32,
        //};
        let sim_controller = RenderSimController::new(render_input.sim);

        let window_size = window.inner_size();
        // The instance is a handle to our GPU
        let instance = wgpu::Instance::default();
        // Safety: lifetime >= window
        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let (adapter, device, queue) = prep_device(instance, &surface).await;
        let (_surface_capabilites, surface_config) =
            prep_surface(&surface, &adapter, window_size, &device);

        let shader: wgpu::ShaderModule =
            device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let vertices = Vertex::get_vertices(render_input.width, render_input.height);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let num_vertices = vertices.len() as u32;

        let trace_colors: Vec<_> = render_input
            .trace_info
            .iter()
            .flat_map(|t| {
                [
                    u32::from_le_bytes(t.color_off),
                    u32::from_le_bytes(t.color_on),
                ]
            })
            .collect();
        let trace_color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&trace_colors),
            usage: wgpu::BufferUsages::STORAGE,
        });

        //let traces: Vec<u32> = render_input.traces.iter().map(|&i| i.into()).collect();
        //let trace_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //    label: None,
        //    contents: bytemuck::cast_slice(&traces),
        //    usage: wgpu::BufferUsages::STORAGE,
        //});
        //let gate_ids = render_input.gate_ids.clone();
        //let gate_id_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //    label: None,
        //    contents: bytemuck::cast_slice(&gate_ids),
        //    usage: wgpu::BufferUsages::STORAGE,
        //});

        //TODO: PERF: reduce this buffer size to actual input size
        let bit_state: Vec<u64> = (0..(1024 * 256)).into_iter().map(|x| x).collect();
        let bit_state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&bit_state),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sim_params = SimParams::default();
        let sim_param_data = sim_params.as_arr().to_vec();
        let sim_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim param buffer"),
            contents: bytemuck::cast_slice(&sim_param_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let packed_data: Vec<u32> = render_input
            .traces
            .into_iter()
            .zip(render_input.gate_ids)
            .map(|(trace, id)| pack_single(id, trace))
            .collect();
        let packed_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Packed data buffer"),
            contents: bytemuck::cast_slice(&packed_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let texture_size = wgpu::Extent3d {
            width: render_input.width.try_into().unwrap(),
            height: render_input.height.try_into().unwrap(),
            depth_or_array_layers: 1,
        };
        let data_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("data texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &data_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All, // ??
            },
            bytemuck::cast_slice(&packed_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(
                    (mem::size_of::<u32>() * render_input.width)
                        .try_into()
                        .unwrap(),
                ),
                rows_per_image: std::num::NonZeroU32::new(render_input.height.try_into().unwrap()),
            },
            texture_size,
        );
        let data_texture_view = data_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let data_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            label: None,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: None,
            border_color: None,
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
                            u64::try_from(sim_param_data.len() * mem::size_of::<f32>()).unwrap(),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
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
                    resource: bit_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sim_param_buffer.as_entire_binding(),
                },
                //wgpu::BindGroupEntry {
                //    binding: 2,
                //    resource: trace_buffer.as_entire_binding(),
                //},
                //wgpu::BindGroupEntry {
                //    binding: 3,
                //    resource: gate_id_buffer.as_entire_binding(),
                //},
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: trace_color_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: packed_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&data_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&data_sampler),
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
            bit_state,
            bit_state_buffer,
            bind_group,
            sim_params,
            sim_param_buffer,
            keystates: KeyStates::default(),
            last_update: Instant::now(),
            sim_controller,
            last_print: Instant::now(),
            state_history: Vec::new(),
            vertex_buffer,
            num_vertices,
            view,
            //trace_buffer,
            //gate_id_buffer,
            //trace_color_buffer,
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
                W | F => k.up = p,
                A | R => k.right = n,
                S => k.up = n,
                D | T => k.right = p,
                J => k.zoom = n,
                K => k.zoom = p,
                P if *state == ElementState::Pressed => k.reverse = !k.reverse,
                _ => (),
            }
        };
        false
    }

    fn update(&mut self) {
        //for y in self.bit_state.iter_mut() {
        //    *y += 1;
        //}
        if self.keystates.reverse {
            if let Some(state) = self.state_history.pop() {
                self.bit_state = state;
            } else {
                self.keystates.reverse = false;
            }
        } else {
            self.state_history.push(mem::take(&mut self.bit_state));
            self.sim_controller.get_state_in(&mut self.bit_state);
        }

        let wsize = self.window.inner_size();
        let ratio = wsize.width as f32 / wsize.height as f32;

        self.sim_params.zoom_y = 1.0 / self.sim_params.zoom_y;
        self.sim_params.zoom_x = 1.0 / self.sim_params.zoom_x;

        let scale = self.sim_params.zoom_y;
        let dt = mem::replace(&mut self.last_update, Instant::now())
            .elapsed()
            .as_secs_f32();
        let ds = scale * dt;
        //let dt = self.last_update.elapsed().as_secs_f32();

        //self.last_update = Instant::now();
        self.sim_params.offset_x -= 4.0 * self.keystates.right * ds;
        self.sim_params.offset_y -= 4.0 * self.keystates.up * ds;

        //let max_dim = self.sim_params.max_y.max(self.sim_params.max_x);
        //let min_zoom = 4.0;

        let dz = self.sim_params.zoom_y * self.keystates.zoom * dt; //
                                                                    //.clamp(
                                                                    //    min_zoom - self.sim_params.zoom_y,
                                                                    //    max_dim - self.sim_params.zoom_y,
                                                                    //);

        self.sim_params.zoom_y += dz;
        //self.sim_params.offset_x += ratio * dz / 2.0;
        //self.sim_params.offset_y += dz / 2.0;

        self.sim_params.zoom_x = self.sim_params.zoom_y * ratio;

        self.sim_params.zoom_y = 1.0 / self.sim_params.zoom_y;
        self.sim_params.zoom_x = 1.0 / self.sim_params.zoom_x;

        self.view.trace_x += dt * self.view.zoom * self.keystates.right;
        self.view.trace_y += dt * self.view.zoom * self.keystates.up;
        self.view.zoom += dt * self.view.zoom * self.keystates.zoom;

        self.sim_params = self.view.as_sim_params(ratio);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = (self.surface.get_current_texture())?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.queue.write_buffer(
            &self.bit_state_buffer,
            0, /* offset */
            bytemuck::cast_slice(&self.bit_state),
        );
        let sim_params_arr = self.sim_params.as_arr();
        self.queue.write_buffer(
            &self.sim_param_buffer,
            0, /* offset */
            bytemuck::cast_slice(&sim_params_arr),
        );

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
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            render_pass.draw(0..self.num_vertices, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn prep_surface(
    surface: &wgpu::Surface,
    adapter: &wgpu::Adapter,
    size: winit::dpi::PhysicalSize<u32>,
    device: &wgpu::Device,
) -> (wgpu::SurfaceCapabilities, wgpu::SurfaceConfiguration) {
    let surface_capabilites = surface.get_capabilities(adapter);

    // Try to get a linear color space
    let surface_format: wgpu::TextureFormat = surface_capabilites
        .formats
        .iter()
        .copied()
        .find(|f| !f.describe().srgb)
        .unwrap_or(surface_capabilites.formats[0]);

    let surface_config: wgpu::SurfaceConfiguration = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::AutoVsync, // surface_capabilites.present_modes[0]
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
            power_preference: wgpu::PowerPreference::HighPerformance, //default(),
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
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[Vertex::desc()],
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
    })
}

pub async fn run<S: RenderSim + Send + 'static>(render_input: RenderInput<S>) {
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
                let frame_print_freq = 60;
                if frame_count == frame_print_freq {
                    let avg_frame_time = accum_time * 1000.0 / frame_count as f32;
                    let (ticks_elapsed, time_elapsed) = state.sim_controller.get_counter();
                    let tps = ticks_elapsed as f32 / time_elapsed.as_secs_f32();

                    let fps = frame_print_freq as f32
                        / mem::replace(&mut state.last_print, Instant::now())
                            .elapsed()
                            .as_secs_f32();
                    println!("Avg frame time {avg_frame_time}ms, TPS: {tps}, FPS: {fps}");
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
