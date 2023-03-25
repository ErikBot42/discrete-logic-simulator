// |\ (-1,3)
// |  \
// |    \
// |      \
// |--------\  (1,1)
// |          \
// |  Window  | \
// |          |   \
// |          |     \
// ------------------- (3,-1)
// (-1,-1)


// (array at host side)
//struct SimParams {
//  deltaT : f32,
//  rule1Distance : f32,
//  rule2Distance : f32,
//  rule3Distance : f32,
//  rule1Scale : f32,
//  rule2Scale : f32,
//  rule3Scale : f32,
//};

struct SimParams {
  max_x: f32,
  max_y: f32,
  offset: vec2<f32>,
  zoom: vec2<f32>,
};

// @group(0) @binding(0) var<uniform> params : SimParams;

@group(0) @binding(0) var<storage, read> state: array<u32>;
@group(0) @binding(1) var<uniform> params : SimParams;
//@group(0) @binding(2) var<storage, read> trace: array<u32>;
//@group(0) @binding(3) var<storage, read> gate_id: array<u32>;
@group(0) @binding(2) var<storage, read> trace_color_rgba: array<u32>;
@group(0) @binding(3) var<storage, read> packed: array<u32>;
@group(0) @binding(4) var packed_texture: texture_2d<u32>;
@group(0) @binding(5) var packed_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    //@location(0) @interpolate(linear) tex_coords: vec2<u32>,
    @location(0) trace_pos: vec2<f32>,
};


@vertex
fn vs_main(
    //@builtin(vertex_index) in_vertex_index: u32,
    model: VertexInput,
) -> VertexOutput {
    
    var out: VertexOutput;
    out.clip_position = vec4<f32>((model.position + params.offset) * params.zoom  , 0.0, 1.0);
    out.trace_pos = model.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    //let packed_data_sample = textureSample(packed_texture, packed_sampler, in.tex_coords);
    

    //let packed_data_sample = textureLoad(packed_texture, in.tex_coords, 0).r;
    //let packed_data_sample = textureLoad(packed_texture, vec2<u32>(u32(in.trace_pos.x), u32(in.trace_pos.y)), 0).r;

    let ipos = vec2<u32>(in.trace_pos);
    let index = u32(ipos.x + u32(params.max_x) * ipos.y);
    let mask = u32(0xFFFFFF);
    let packed_data: u32 = packed[index];
    let gate: u32 = packed_data & mask;
    let t = packed_data >> u32(24);
    let is_on: u32 = (state[gate / u32(32)] >> (gate % u32(32))) & u32(1);
    var rgba = trace_color_rgba[is_on + t * u32(2)];

    // rgba -> vec4<f32>
    let ri = rgba & u32(0xFF); rgba >>= u32(8);
    let gi = rgba & u32(0xFF); rgba >>= u32(8);
    let bi = rgba & u32(0xFF); rgba >>= u32(8);
    //let ai = rgba & u32(0xFF); rgba >>= u32(8);

    let r = f32(ri) / f32(255.0);
    let g = f32(gi) / f32(255.0);
    let b = f32(bi) / f32(255.0);

    return vec4<f32>(r, g, b, 1.0);
    //return vec4<f32>(dpdx(r), dpdx(g), dpdx(b), 1.0);
}
