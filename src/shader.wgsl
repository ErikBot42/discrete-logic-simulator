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
  offset_x: f32,
  offset_y: f32,
  zoom_x: f32,
  zoom_y: f32,
};

// @group(0) @binding(0) var<uniform> params : SimParams;

@group(0) @binding(0) var<storage, read> state: array<u32>;
@group(0) @binding(1) var<uniform> params : SimParams;
@group(0) @binding(2) var<storage, read> trace: array<u32>;
@group(0) @binding(3) var<storage, read> gate_id: array<u32>;
@group(0) @binding(4) var<storage, read> trace_color_rgba: array<u32>;
@group(0) @binding(5) var<storage, read> packed: array<u32>;


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) trace_pos: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var x: f32;
    var y: f32;
    switch in_vertex_index {
        case 0u: {
            x = 3.0; // right
            y = -1.0;
        }
        case 1u: {
            x = -1.0; // top
            y = 3.0;
        }
        default { // 2
            x = -1.0; // left
            y = -1.0;
        }
    }
    var out: VertexOutput;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);

    let normalized_pos = (vec2<f32>(x, y) + vec2<f32>(1.0, 1.0))/vec2<f32>(2.0, 2.0);
    out.trace_pos = normalized_pos;
    out.trace_pos.x *= params.zoom_x;
    out.trace_pos.y *= params.zoom_y;
    out.trace_pos -= vec2<f32>(params.offset_x, params.offset_y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ipos = vec2<i32>(in.trace_pos);
    let ilimit = vec2<i32>(i32(params.max_x), i32(params.max_y));
    if ipos.x <= 0 || ipos.y <= 0 || ilimit.x < ipos.x || ilimit.y < ipos.y {
        return vec4<f32>(0.1,0.1,0.1,1.0);
    } else {
        let index = u32(ipos.x + ilimit.x * ipos.y);
        //let gate: u32 = gate_id[index];
        //let t = trace[index];
        
        let mask = u32(0xFFFFFF);
        let packed_data: u32 = packed[index];
        let gate: u32 = packed_data & mask;
        let t = packed_data >> u32(24);

        let is_on: u32 = (state[gate / u32(32)] >> (gate % u32(32))) & u32(1);

        var rgba = trace_color_rgba[is_on + t * u32(2)];
        let ri = rgba & u32(0xFF); rgba >>= u32(8);
        let gi = rgba & u32(0xFF); rgba >>= u32(8);
        let bi = rgba & u32(0xFF); rgba >>= u32(8);
        let ai = rgba & u32(0xFF); rgba >>= u32(8);

        let r = f32(ri) / f32(255.0);
        let g = f32(gi) / f32(255.0);
        let b = f32(bi) / f32(255.0);

        return vec4<f32>(r, g, b, 1.0);
    }
}
