
    use super::*;
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
            let other_id =
                unwrap_or_else!(other.id, { add_new_id(elements, nodes, width, other_x) });
            add_connection(nodes, (this_id, other_id), dir);
        }
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
                let other_kind =
                    unwrap_or_else!(elements.get(other_x as usize), continue 'side).trace;
                if other_kind == Trace::Cross {
                    continue 'forward;
                }
                if other_kind.is_same_as(this_kind) {
                    fill_id(elements, width, other_x, id);
                }
                continue 'side;
            }
        }
    }

    // this HAS to be recursive since gates have arbitrary shapes.
    #[inline]
    pub(super) fn explore(
        elements: &mut Vec<BoardElement>,
        nodes: &mut Vec<BoardNode>,
        width: i32,
        this_x: i32,
    ) {
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
