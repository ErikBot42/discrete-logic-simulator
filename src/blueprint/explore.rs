use super::*;

/// Represents one gate or trace
pub(super) struct BoardNode {
    inputs: BTreeSet<usize>,
    outputs: BTreeSet<usize>,
    kind: GateType,
    initial_state: bool,
    pub(crate) network_id: Option<usize>,
}
impl BoardNode {
    #[must_use]
    pub(super) fn new(trace: Trace) -> Self {
        let (kind, initial_state) = trace.to_gatetype_state();
        BoardNode {
            inputs: BTreeSet::new(),
            outputs: BTreeSet::new(),
            initial_state,
            kind,
            network_id: None,
        }
    }
}

pub(super) fn compile_network<T: LogicSim>(
    plain: &parse::VcbPlainBoard,
) -> (usize, usize, Vec<BoardNode>, Vec<BoardElement>, GateNetwork) {
    let height = plain.height;
    let width = plain.width;
    let num_elements = width * height;
    let mut nodes = Vec::new();
    let elements = timed!(
        {
            let mut elements: Vec<_> = plain
                .traces
                .iter()
                .cloned()
                .map(BoardElement::new)
                .collect();
            for x in 0..num_elements {
                explore(
                    &mut elements,
                    &mut nodes,
                    width.try_into().unwrap(),
                    x.try_into().unwrap(),
                );
            }
            elements
        },
        "create elements: {:?}"
    );
    let network = {
        let mut network = GateNetwork::default();

        // add vertexes to network
        for node in &mut nodes {
            node.network_id = Some(network.add_vertex(node.kind, node.initial_state));
        }
        // add edges to network
        for (i, node) in nodes.iter().enumerate() {
            for input in &node.inputs {
                assert!(nodes[*input].outputs.contains(&i));
            }
            let mut inputs: Vec<usize> = node
                .inputs
                .clone()
                .into_iter()
                .map(|x| nodes[x].network_id.unwrap())
                .collect();
            inputs.sort_unstable();
            inputs.dedup();
            network.add_inputs(node.kind, node.network_id.unwrap(), inputs);
        }
        network
    };
    (height, width, nodes, elements, network)
}

mod explore_new {
    // stack search floodfill
    // of other has ID, connect, otherwise, noop
    use crate::blueprint::Trace;
    use std::collections::HashMap;
    use Trace::*;

    // We are ignoring the Random trace since it is non deterministic.
    fn is_passive(trace: Trace) -> bool {
        matches!(trace, Empty | Annotation | Filler | Random)
    }
    #[rustfmt::skip]
    // INCLUDES READ/WRITE
    fn is_wire(trace: Trace) -> bool {
        matches!(
            trace, 
            Gray | White | Red | Orange1 | Orange2 | Orange3 | Yellow | Green1 | Green2 | Cyan1 | Cyan2 | Blue1 | Blue2 | Purple | Magenta | Pink | Read | Write
        )
    }
    fn is_bus(trace: Trace) -> bool {
        matches!(
            trace,
            BusRed | BusGreen | BusBlue | BusTeal | BusPurple | BusYellow
        )
    }

    /// is it valid for this gate to have outputs through write
    #[rustfmt::skip]
    fn can_have_outputs(trace: Trace) -> bool {
        matches!(
            trace,
            Vmem | Buffer | And | Or | Xor | Not | Nand | Nor | Xnor | LatchOn | LatchOff | Led | Wireless0 | Wireless1 | Wireless2 | Wireless3 | Timer | Break | Clock
        )
    }
    /// is it valid for this gate to have inputs through read
    #[rustfmt::skip]
    fn can_have_inputs(trace: Trace) -> bool {
        matches!(
            trace,
            Vmem | Buffer | And | Or | Xor | Not | Nand | Nor | Xnor | LatchOn | LatchOff | Led | Wireless0 | Wireless1 | Wireless2 | Wireless3
        )
    }

    fn index((x, y): (i32, i32), width: i32, height: i32) -> Option<usize> {
        (x >= 0 && y >= 0 && x < width && y < height).then_some((x + y * width) as usize)
    }

    fn should_merge(this: Trace, other: Trace) -> bool {
        this == other || (is_wire(this) && is_wire(other)) || (is_bus(this) || is_bus(other))
    }

    fn parse(traces: Vec<Trace>, width: usize, height: usize) {
        /*match traces[0] {
            Write => (), standard
            Read => (), standard

            Empty => (), not logic
            Annotation => (),
            Filler => (),

            Cross => (), traversal
            Tunnel => (),


            Wireless0 => (),
            Wireless1 => (),
            Wireless2 => (),
            Wireless3 => (),

            Vmem => (), -> latch

            Mesh => (), -> special case, can contain buses

            Timer => (),
            Clock => (),

            Random => (),

            Break => (),

            model buses as OR-gates
        }*/
        // (network id of bus, trace) -> network id of trace
        // NOTE: Same id for diffrent traces possible.

        //let timer_id: Option<usize> = None;
        //let clock_id: Option<usize> = None;
        //let break_id: Option<usize> = None;
        //let wireless_id: [Option<usize>; 4] = [None; 4];
        //let mesh_id: Option<usize> = None;

        //let bus_connections: HashMap<(usize, Trace), usize> = HashMap::new();
        //let mesh_connections: HashMap<Trace, usize> = HashMap::new();

        // fill these first, then handle other cases.

        let mut fill_state = FillState::default();
        let mut connect_state = ConnectState::default();

        // trace id -> gate id
        // let mut trace_index_map: Vec<Option<usize>> = Vec::new();

        let mut max_trace_id = 0;

        let mut ids: Vec<Option<usize>> = (0..traces.len()).map(|_| None).collect();

        fun_name(
            height,
            width,
            &mut ids,
            &traces,
            connect_state,
            fill_state,
            &mut max_trace_id,
        );
    }

    fn fun_name(
        height: usize,
        width: usize,
        mut ids: &mut [Option<usize>],
        traces: &[Trace],
        mut connect_state: ConnectState,
        mut fill_state: FillState,
        max_trace_id: &mut usize,
    ) {
        let mut front: Vec<((i32, i32), usize)> = Vec::new();
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let pos = (x, y);
                explore(
                    pos,
                    &mut ids,
                    width as i32,
                    height as i32,
                    &traces,
                    None,
                    &mut front,
                    &mut connect_state,
                    &mut fill_state,
                    max_trace_id,
                );

                // TODO: faster floodfill
                while let Some((pos, prev_id)) = front.pop() {
                    explore(
                        pos,
                        &mut ids,
                        width as i32,
                        height as i32,
                        &traces,
                        Some(prev_id),
                        &mut front,
                        &mut connect_state,
                        &mut fill_state,
                        max_trace_id,
                    );
                }
            }
        }
    }
    #[derive(Default)]
    struct FillState {
        timer_id: Option<usize>,         // handle at floodfill time
        clock_id: Option<usize>,         // handle at floodfill time
        break_id: Option<usize>,         // handle at floodfill time
        mesh_id: Option<usize>, // does not need id, but needs same default id, to delete it
        wireless_id: [Option<usize>; 4], // needs same id, set at floodfill time
    }
    fn explore(
        pos: (i32, i32),
        ids: &mut [Option<usize>],
        width: i32,
        height: i32,
        traces: &[Trace],
        prev_id: Option<usize>,
        front: &mut Vec<((i32, i32), usize)>,
        connect_state: &mut ConnectState,
        fill_state: &mut FillState,
        max_trace_id: &mut usize,
    ) {
        let dim = (width, height);
        let this_idx = unwrap_or_else!(index(pos, width, height), return);
        let this_trace = traces[this_idx];
        if ids[this_idx].is_some() || is_passive(this_trace) {
            return;
        }
        // this trace has no id, lets add a new id.
        let this_id = prev_id.unwrap_or_else(|| {
            let mut none = None;
            *match this_trace {
                Clock => &mut fill_state.clock_id,
                Mesh => &mut fill_state.mesh_id,
                Timer => &mut fill_state.timer_id,
                Break => &mut fill_state.break_id,
                Wireless0 => &mut fill_state.wireless_id[0],
                Wireless1 => &mut fill_state.wireless_id[1],
                Wireless2 => &mut fill_state.wireless_id[2],
                Wireless3 => &mut fill_state.wireless_id[3],
                _ => &mut none,
            }
            .get_or_insert_with(|| std::mem::replace(max_trace_id, *max_trace_id + 1))
        });
        ids[this_idx] = Some(this_id);

        for delta in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let (new_pos, new_idx, new_trace) =
                unwrap_or_else!(index_offset(pos, delta, 1, dim, traces), continue);
            if should_merge(this_trace, new_trace) {
                if ids[new_idx].is_none() {
                    front.push((new_pos, this_id));
                }
            } else if new_trace == Trace::Cross {
                let (new_pos, new_idx, new_trace) =
                    unwrap_or_else!(index_offset(pos, delta, 2, dim, traces), continue);
                if should_merge(this_trace, new_trace) {
                    if ids[new_idx].is_none() {
                        front.push((new_pos, this_id));
                    }
                }
            } else if new_trace == Trace::Tunnel {
                // start iteration at 2, single tunnel not valid.
                for dd in 2.. {
                    let (_, _, new_trace) =
                        unwrap_or_else!(index_offset(pos, delta, dd, dim, traces), break);
                    if new_trace == Trace::Tunnel {
                        let (new_pos, new_idx, new_trace) =
                            unwrap_or_else!(index_offset(pos, delta, dd + 1, dim, traces), break);
                        if new_trace == this_trace {
                            if ids[new_idx].is_none() {
                                front.push((new_pos, this_id));
                            }
                            break;
                        }
                    }
                }
            } else if new_trace.is_logic() {
                connect((this_idx, this_trace), (new_idx, new_trace), connect_state);
                connect((new_idx, new_trace), (this_idx, this_trace), connect_state);
            }
        }
    }

    #[derive(Default)]
    struct ConnectState {
        // need dynamic construction.
        connections: Vec<(usize, usize)>, // sort, dedup after merge
        mesh_connections: HashMap<Trace, Vec<usize>>,
        bus_connections: HashMap<(usize, Trace), Vec<usize>>,
    }
    // connect THIS -> OTHER
    fn connect(
        (this_idx, this_trace): (usize, Trace),
        (other_idx, other_trace): (usize, Trace),
        state: &mut ConnectState,
    ) {
        //let mut outputs: Vec<Vec<usize>> = Vec::new(); // need to dedup later

        //let mut mesh_connections: HashMap<Trace, Vec<usize>> = HashMap::new();
        // (bus trace id, trace) -> trace id
        //let mut bus_connections: HashMap<(usize, Trace), Vec<usize>> = HashMap::new();

        if other_trace == Trace::Mesh {
            state
                .mesh_connections
                .entry(this_trace)
                .or_default()
                .push(this_idx);
            return;
        }

        if is_bus(other_trace) {
            state
                .bus_connections
                .entry((other_idx, this_trace))
                .or_default()
                .push(this_idx);
            return;
        }

        // this -> other
        if can_have_outputs(this_trace) && other_trace == Trace::Write {
            state.connections.push((this_idx, other_idx));
        }
        // other -> this
        if can_have_inputs(this_trace) && other_trace == Trace::Read {
            state.connections.push((other_idx, this_idx));
        }
    }
    fn index_offset(
        (pos_x, pos_y): (i32, i32),
        (dx, dy): (i32, i32),
        dd: i32,
        (width, height): (i32, i32),
        traces: &[Trace],
    ) -> Option<((i32, i32), usize, Trace)> {
        let new_pos = (pos_x + dx * dd, pos_y + dy * dd);
        index(new_pos, width, height).map(|idx| (new_pos, idx, traces[idx]))
    }
}

// this HAS to be recursive since gates have arbitrary shapes.
#[inline]
fn explore(elements: &mut Vec<BoardElement>, nodes: &mut Vec<BoardNode>, width: i32, this_x: i32) {
    // if prev_id is Some, merge should be done.
    // don't merge if id already exists, since that wouldn't make sense

    let this = &elements[TryInto::<usize>::try_into(this_x).unwrap()];
    let this_kind = this.trace;

    if !this_kind.is_logic() {
        return;
    }

    let this_id = unwrap_or_else!(this.id, { add_new_id(elements, nodes, width, this_x) });

    assert!(this_kind.is_logic());
    if !this_kind.is_gate() {
        return;
    }

    connect_id(elements, nodes, width, this_x, this_id);
}

// pre: this trace is logic
#[must_use]
fn add_new_id(
    elements: &mut Vec<BoardElement>,
    nodes: &mut Vec<BoardNode>,
    width: i32,
    this_x: i32,
) -> usize {
    let this = &elements[TryInto::<usize>::try_into(this_x).unwrap()];
    assert!(this.trace.is_logic());

    // create a new id.
    let this_id = nodes.len();
    nodes.push(BoardNode::new(this.trace));

    // fill with this_id
    fill_id(elements, width, this_x, this_id);
    this_id
}

/// floodfill with `id` at `this_x`
/// pre: logic trace, otherwise id could have been
/// assigned to nothing, `this_x` valid
fn fill_id(elements: &mut Vec<BoardElement>, width: i32, this_x: i32, id: usize) {
    let this = &mut elements[this_x as usize];
    let this_kind = this.trace;
    assert!(this_kind.is_logic());
    match this.id {
        None => this.id = Some(id),
        Some(this_id) => {
            assert_eq!(this_id, id);
            return;
        },
    }
    'side: for dx in [1, -1, width, -width] {
        'forward: for ddx in [1, 2] {
            let other_x = this_x + dx * ddx;
            if this_x % width != other_x % width && this_x / width != other_x / width {
                continue 'side; // prevent wrapping connections
            }
            let other_kind = unwrap_or_else!(elements.get(other_x as usize), continue 'side).trace;
            if other_kind == Trace::Cross {
                continue 'forward;
            }
            if other_kind.will_connect(this_kind) {
                fill_id(elements, width, other_x, id);
            }
            continue 'side;
        }
    }
}

// Connect gate with immediate neighbor
// pre: this is gate, this_x valid, this has id.
fn connect_id(
    elements: &mut Vec<BoardElement>,
    nodes: &mut Vec<BoardNode>,
    width: i32,
    this_x: i32,
    this_id: usize,
) {
    let this = &mut elements[TryInto::<usize>::try_into(this_x).unwrap()];
    let this_kind = this.trace;
    assert!(this_kind.is_gate());
    assert!(this.id.is_some());

    'side: for dx in [1, -1, width, -width] {
        let other_x = this_x + dx;
        if this_x % width != other_x % width && this_x / width != other_x / width {
            continue 'side; // prevent wrapping connections
        }
        let other = unwrap_or_else!(elements.get(other_x as usize), continue 'side);
        let dir = match other.trace {
            Trace::Read => false,
            Trace::Write => true,
            _ => continue,
        };
        let other_id = unwrap_or_else!(other.id, { add_new_id(elements, nodes, width, other_x) });
        add_connection(nodes, (this_id, other_id), dir);
    }
}

fn add_connection(nodes: &mut [BoardNode], connection: (usize, usize), swp_dir: bool) {
    let (start, end) = if swp_dir {
        (connection.1, connection.0)
    } else {
        connection
    };
    assert!(start != end);
    let a = nodes[start].inputs.insert(end);
    let b = nodes[end].outputs.insert(start);
    assert!(a == b);
    assert_ne!(
        nodes[start].kind == GateType::Cluster,
        nodes[end].kind == GateType::Cluster
    );
}
