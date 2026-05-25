# Inter-tile communications in KTIR

**Scope:** Four ops — `ktdp.inter_tile_produce`, `ktdp.inter_tile_consume`,
`ktdp.inter_tile_reduce`, and `ktdp.inter_tile_reduce_scatter` —
that together cover all three inter-tile communication patterns: broadcast,
all-reduce, and reduce-scatter.

---

## 1. Motivation

Inter-tile communication involves two orthogonal concerns:

1. **Production** — which tiles contribute data and what they contribute.
2. **Delivery** — how the contributed data is delivered to the receiving
   tiles: pass-through unchanged (broadcast), folded by a combiner
   (reduce), or folded then scattered (reduce-scatter).

Separating these concerns into a unified production op plus three
delivery ops keeps each op single-purpose and enables any combination:

| Pattern | Production op | Delivery op |
|---------|--------------|-------------|
| Broadcast | `ktdp.inter_tile_produce` | `ktdp.inter_tile_consume` |
| Reduce | `ktdp.inter_tile_produce` | `ktdp.inter_tile_reduce` |
| Reduce-scatter | `ktdp.inter_tile_produce` | `ktdp.inter_tile_reduce_scatter` |

`ktdp.inter_tile_produce` returns a `!ktdp.tile_future<T_p>` SSA value.
Each delivery op takes that future as its operand. The def-use edge from
production to delivery encodes the happens-before ordering with no
explicit barriers in the IR. Corresponding production and delivery ops are expected to be adjacent in a single basic block to avoid dead locks.

---

## 2. `ktdp.inter_tile_produce` — unified production op

### 2.1 Attributes

**`producer_tiles_per_group`** — parameterized affine integer set `(i)[g]`
selecting which tiles produce per group. The set has one dimension (the tile
id) and one symbol (`g`, the group index). For example,
`affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>` selects tile ids
`4g .. 4g+3` for any group index `g`. An enumerated form (a list of tile-id
lists) is supported as a fallback when per-group membership is irregular.
- Broadcast: selects exactly one tile per group.
- Reduce / reduce-scatter: selects all tiles in the group.

**Disjointness invariant.** For any two distinct group indices `g_1 != g_2`
in `groups`, `producer_tiles_per_group(g_1)` and
`producer_tiles_per_group(g_2)` must be disjoint. Every producing tile is
in exactly one group. The verifier enforces this. The motivation is
unambiguous group membership: each tile contributes to exactly one group's
production.

**`groups`** — affine integer set defining the range of valid group indices.
For example, `affine_set<(g) : (g >= 0, -g + 7 >= 0)>` defines 8 groups,
indexed `0..7`. Identical in role to the same attribute on the existing ops.

### 2.2 Producer region

The producer region indicates **what partial results each tile contributes**.
It is the per-tile boundary that names the SSA values entering the cross-tile
communication. The block runs once per participating tile, in that tile's
SPMD execution.

**Block argument:** `%gid: index` — the index of the group this tile
belongs to. The runtime binding is direct: tile `t` finds its group by
looking up which entry of `producer_tiles_per_group(g)` contains `t`;
that `g` is bound to `%gid` for tile `t`'s execution of the block.

The body knows its tile id via `ktdp.get_compute_tile_id` (the same way
every SPMD KTIR body does) and its group index via `%gid`.

**Termination:** the block terminates with
`ktdp.yield_partial %val_1, ..., %val_N : T_p_1, ..., T_p_N`, yielding
one value per partial-tensor role. The yielded values may reference SSA
values from the enclosing scope — typical use is a thin contribution
marker:

```mlir
{
  ^bb0(%gid: index):
    ktdp.yield_partial %my_partial_1, ..., %my_partial_N
                       : T_p_1, ..., T_p_N
}
```

with the per-tile compute that produced `%my_partial` living at function
scope (where it is naturally executed by every tile under SPMD). Richer
bodies are allowed when the contribution is awkward to hoist — e.g.,
compute that depends only on `%gid`, or visibly local "contribution
preparation" the author wants to keep adjacent to the op.

### 2.3 Op signature

```mlir
%future = ktdp.inter_tile_produce
    producer_tiles_per_group = <affine-set>,
    groups                   = <affine-set>
    : T_p_1, ..., T_p_N -> !ktdp.tile_future<T_p_1, ..., T_p_N>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

`%future` is a workgroup-visible handle. It cannot materialize until all
producer tiles in each group have executed the block and terminated with
`ktdp.yield_partial`.

---

## 3. `ktdp.inter_tile_consume` — plain delivery op (broadcast)

### 3.1 Operand and attributes

**Operand:** `!ktdp.tile_future<T_p>` — the future returned by the
corresponding `ktdp.inter_tile_produce`. The def-use edge is the ordering
constraint.

**`consumer_tiles_per_group`** — tiles that receive the delivered value
per group.

**`groups`** — must match the corresponding `inter_tile_produce`.

### 3.2 Semantics

No combining occurs. The value produced by the producer tile(s) in each
group is delivered unchanged to every consumer tile in that group.
When `producer_tiles_per_group` selects one tile per group, every
consumer tile in that group receives the same value (broadcast semantics).

The operations that use `%result` are performed only by the tiles specified
by `consumer_tiles_per_group`. This ownership constraint is enforced by
the def-use chain from `%result`: any use of `%result` is reachable only
by consumer tiles.

No block is needed — post-delivery computation is ordinary function-scope
SPMD code that uses the SSA value.

### 3.3 Op signature

```mlir
%result = ktdp.inter_tile_consume(%future)
    consumer_tiles_per_group = <affine-set>,
    groups                   = <affine-set>
    : !ktdp.tile_future<T_p> -> T_p
```

---

## 4. `ktdp.inter_tile_reduce` — reduction delivery op

### 4.1 Operand and attributes

**Operand:** `!ktdp.tile_future<T_p>` returned by
`ktdp.inter_tile_produce`.

**`consumer_tiles_per_group`** — tiles that receive the reduced result.

**`groups`** — must match the corresponding `inter_tile_produce`.

**`identity`** — N variadic SSA operands, one per partial-tensor role.
Each identity tensor's shape and element type must match the corresponding
partial type `T_p_i` (not the rank-reduced result type `T_r_i`). The
identities are hoisted before the op and shared across all groups and all
tiles. Combining any identity with its corresponding partial yields that
partial.

### 4.2 Reducer region

The op has a single region with a block that receives `2N` arguments —
`%lhs_1, ..., %lhs_N, %rhs_1, ..., %rhs_N` with each `%lhs_i` and
`%rhs_i` of type `T_p_i` — and terminates with
`ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N`.

**Purity.** The combiner must be pure — no memory effects, no calls to
side-effecting ops. Pure tensor ops (`tensor.empty`, `linalg` on tensors,
`arith.*`) are allowed. The verifier rejects combiners containing ops with
side effects.

**Combine ordering.** The associative-commutative contract is by user
agreement; the scheduler is free to combine in tree, ring, linear, or any
hardware-native topology. Different groups' reductions are independent and
may be scheduled in parallel.

### 4.3 Type rules

For each role `i`, `T_r_i` is `T_p_i` with the within-group tile axes
collapsed. The same set of axes is removed for all roles.

### 4.4 Op signature

```mlir
%r_1, ..., %r_N = ktdp.inter_tile_reduce(%future)
    consumer_tiles_per_group = <affine-set>,
    groups                   = <affine-set>,
    identity(%id_1 : T_p_1, ..., %id_N : T_p_N)
    : !ktdp.tile_future<T_p_1, ..., T_p_N> -> T_r_1, ..., T_r_N
{
  ^bb0(%lhs_1: T_p_1, ..., %lhs_N: T_p_N,
       %rhs_1: T_p_1, ..., %rhs_N: T_p_N):
    ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

### 4.5 Result semantics

The op produces N variadic SSA values, one per partial-tensor role. The
values are *per-tile-valued* — each consumer tile holds a result value
when the op completes. Every consumer tile in a group holds the same
value — that group's fully reduced result. Tiles in different groups hold
different values (each its own group's reduction).

**Non-participating tiles.** Results are undefined for tiles not in `consumer_tiles_per_group`.

**Multi-tensor (variadic) reductions.** N ≥ 1 partials are supported.
Argmax-style reductions, where each partial is a correlated tuple of
tensors (values, indices), use N = 2: two identities, two yielded
partials, four reducer region arguments yielding two combined values,
two op results. The structure generalizes uniformly.

---

## 5. `ktdp.inter_tile_reduce_scatter` — reduction + scatter delivery op

### 5.1 Operand and attributes

Identical to §4.1, plus one additional attribute:

**`scatter_dimension`** (i64) — axis of the post-reduction type along
which the result is split row-major across the consumer tiles. The size
along this dimension must be divisible by the per-group consumer-tile
count.

**Per-tile slice.** For a tile with within-group local index `l` (its
ascending position among the consumer tiles in the group), the slice
received is `reduced[l*chunk : (l+1)*chunk]` along `scatter_dimension`,
where `chunk = post_reduction_shape[scatter_dimension] / |consumer tiles per group|`.

### 5.2 Reducer region

Identical to §4.2: pure, associative-commutative, combine ordering
unspecified. Different groups' reductions proceed independently.

### 5.3 Type rules

For each role `i`, `T_r_i` is `T_p_i` with the within-group tile axes
collapsed and then sliced along `scatter_dimension`. The same axes and the
same scatter split apply to all roles.

### 5.4 Op signature

```mlir
%chunk_1, ..., %chunk_N = ktdp.inter_tile_reduce_scatter(%future)
    consumer_tiles_per_group = <affine-set>,
    groups                   = <affine-set>,
    scatter_dimension        = <i64>,
    identity(%id_1 : T_p_1, ..., %id_N : T_p_N)
    : !ktdp.tile_future<T_p_1, ..., T_p_N> -> T_r_1, ..., T_r_N
{
  ^bb0(%lhs_1: T_p_1, ..., %lhs_N: T_p_N,
       %rhs_1: T_p_1, ..., %rhs_N: T_p_N):
    ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

### 5.5 Result semantics

Each consumer tile in a group holds its own row-major slice of that
group's reduced result along `scatter_dimension`. Different tiles in the
same group hold different non-overlapping slices whose concatenation is
the group's full reduced result. Tiles in different groups hold results
from their respective independent reductions.

**Non-participating tiles.** Same constraint as §4.5.

---

## 6. Synchronization model

No explicit barriers appear in the IR. The `!ktdp.tile_future<T_p>` SSA
value encodes two ordering constraints:

1. `%future` cannot materialize until all producer tiles in each group
   have completed their block.
2. The delivery op (`inter_tile_consume` / `inter_tile_reduce` /
   `inter_tile_reduce_scatter`) cannot execute until `%future` is defined.

In SPMD KTIR, a tile cannot observe other tiles' partials except through a
dialect-defined boundary. The `ktdp.inter_tile_produce` block is that
boundary — it names the per-tile contribution, and the
`!ktdp.tile_future` type enforces that delivery ops can only execute after
all producers in the group have completed. The delivery op's result tensor
is an SSA value that cannot materialize until the cross-tile operation
completes; standard MLIR dataflow ordering applies.

Lowering inserts target-specific hardware barriers.

---

## 7. Coverage of inter-core communication patterns

These four ops are sufficient to express all three inter-core
communication patterns:

| Pattern | `inter_tile_produce` | Delivery op | `scatter_dimension` |
|---------|---------------------|-------------|---------------------|
| Broadcast | one producer tile per group | `inter_tile_consume` | absent |
| Reduce | all tiles per group | `inter_tile_reduce` | absent |
| Reduce-scatter | all tiles per group | `inter_tile_reduce_scatter` | present |

---

## 8. Pattern instantiation

### 8.1 Broadcast  →  `inter_tile_produce` + `inter_tile_consume`

```mlir
// 4 tiles, 1 group: tile 0 loads W; all 4 tiles compute.
#tile_0          = affine_set<(i)[g] : (i - 4*g == 0)>
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#single_group    = affine_set<(g) : (g == 0)>

%W_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #tile_0,
    groups                   = #single_group
    : tensor<64x128xf16> -> !ktdp.tile_future<tensor<64x128xf16>>
{
  ^bb0(%gid: index):
    %W = ktdp.load ...
    ktdp.yield_partial %W : tensor<64x128xf16>
}

// Every consumer tile extracts its copy; no combiner → value passes through.
%W_tile = ktdp.inter_tile_consume(%W_future)
    consumer_tiles_per_group = #all_group_tiles,
    groups                   = #single_group
    : !ktdp.tile_future<tensor<64x128xf16>> -> tensor<64x128xf16>

// Post-delivery SPMD compute — owned by consumer_tiles_per_group.
// Ownership verified by traversing the def-use chain from %W_tile.
%A = ktdp.load ...
%C = linalg.matmul ins(%A, %W_tile ...) ...
ktdp.store %C, ...
```

### 8.2 Reduce  →  `inter_tile_produce` + `inter_tile_reduce`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial; future carries all partials.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles,
    groups                   = #all_groups
    : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_2d : tensor<1x64xf16>
}

// Reduce all partials; every consumer tile receives the same reduced value.
%reduced = ktdp.inter_tile_reduce(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    groups                   = #all_groups,
    identity(%add_id : tensor<1x64xf16>)
    : !ktdp.tile_future<tensor<1x64xf16>> -> tensor<1x64xf16>
{
  ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
    %sum = linalg.add ins(%lhs, %rhs ...) ...
    ktdp.yield_reduced %sum : tensor<1x64xf16>
}
```

#### 8.2.1 Full IR — single-group reduce (96×64)

**Layout and partitioning.** `A` and `B` are `tensor<96x64xf16>` in HBM.
The kernel computes the column-wise sum of `A + B`, producing a
1-D `tensor<64xf16>`.

The 32 compute tiles form a single group. Tile `t` owns rows
`t*3 .. t*3+2` of `A` and `B` — a 3×64 slab each. The per-tile
contribution is the row-reduced partial expanded to `tensor<1x64xf16>`,
where the leading unit dimension is the within-group tile axis the op
collapses. Every tile holds the same `%reduced : tensor<64xf16>` (all-reduce case: consumer set = producer set).

```mlir
#A_view_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#AB_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 +  2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#E_view_set  = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#E_tile_set  = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#identity_2d = affine_map<(d0, d1) -> (d0, d1)>

// One group containing all 32 tiles.
#group_tiles = affine_set<(i)[g] : (i - 32*g >= 0, -i + 32*(g+1) - 1 >= 0)>
#all_groups  = affine_set<(g) : (g == 0)>

module {
  func.func @inter_tile_reduce_single_group() {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 3 : index
    %A_start = arith.constant 1024  : index
    %B_start = arith.constant 12288 : index
    %E_start = arith.constant 22528 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Identity: tensor<1x64xf16> of zeros — matches partial type T_p.
    %c_zero   = arith.constant 0.0 : f16
    %add_init = tensor.empty() : tensor<1x64xf16>
    %add_id   = linalg.fill ins(%c_zero : f16) outs(%add_init : tensor<1x64xf16>)
                  -> tensor<1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %t, %tile_size : index

    %A_access = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_2d
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_2d
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    %A_tile = ktdp.load %A_access : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>
    %B_tile = ktdp.load %B_access : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>

    %AB_init = tensor.empty() : tensor<3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile : tensor<3x64xf16>, tensor<3x64xf16>)
                          outs(%AB_init : tensor<3x64xf16>) -> tensor<3x64xf16>

    %red_init   = tensor.empty() : tensor<64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16) outs(%red_init : tensor<64xf16>)
                    -> tensor<64xf16>
    %partial_1d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<3x64xf16>)
                    outs(%red_filled : tensor<64xf16>)
                    dimensions = [0]
    %partial_2d = tensor.expand_shape %partial_1d [[0, 1]] output_shape [1, 64]
                    : tensor<64xf16> into tensor<1x64xf16>

    // Produce: every tile contributes its partial_2d to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles,
        groups                   = #all_groups
        : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_2d : tensor<1x64xf16>
    }

    // Reduce: unit dim 0 is the within-group tile axis; the op collapses it.
    // Every tile holds the same %reduced : tensor<64xf16> (all-reduce case).
    %reduced = ktdp.inter_tile_reduce(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        groups                   = #all_groups,
        identity(%add_id : tensor<1x64xf16>)
        : !ktdp.tile_future<tensor<1x64xf16>> -> tensor<64xf16>
    {
      ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
        %init = tensor.empty() : tensor<1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                           outs(%init : tensor<1x64xf16>) -> tensor<1x64xf16>
        ktdp.yield_reduced %sum : tensor<1x64xf16>
    }

    // Post-reduction: every tile redundantly writes the same value.
    %reduced_2d = tensor.expand_shape %reduced [[0, 1]] output_shape [1, 64]
                    : tensor<64xf16> into tensor<1x64xf16>

    %E_view = ktdp.construct_memory_view %E_start, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<1x64xf16>
    %E_access = ktdp.construct_access_tile %E_view[%c0, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_2d
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>

    ktdp.store %reduced_2d, %E_access
              : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

    return
  }
}
```

#### 8.2.2 Full IR — multi-group reduce (128×8×12×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x12x64xf16>` in
HBM. The four axes have distinct roles:

- Dim 0 (size 128): preserved through this op.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 12): the **reduction axis** — within each group, 4 tiles
  cooperate over this axis.
- Dim 3 (size 64): vector / stick axis, preserved.

There are 32 compute tiles forming 8 groups of 4. For tile `t`,
`g = t / 4` and `l = t % 4`. Tile `(g, l)` reads slice
`[*, g, l*3 : l*3+3, *]` of `A` and `B` — shape `<128x1x3x64>` each.

The partial is `<128x1x1x64>`: dim 1 is the group axis (preserved), dim 2
is the within-group tile axis (collapsed by the op to `<128x1x64>`). All
four tiles in a group hold identical values; different groups hold
different values.

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 2   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#E_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 3   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#E_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 == 0,
     d3 >= 0, -d3 + 63 >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_reduce_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %red_slab = arith.constant 3 : index   // 12 / 4

    %A_start = arith.constant 1024     : index
    %B_start = arith.constant 12583936 : index
    %E_start = arith.constant 25166848 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x12x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x12x64xf16>

    // Identity: tensor<128x1x1x64xf16> of zeros — matches partial type T_p.
    %c_zero  = arith.constant 0.0 : f16
    %id_init = tensor.empty() : tensor<128x1x1x64xf16>
    %add_id  = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<128x1x1x64xf16>)
                 -> tensor<128x1x1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %red_anchor = arith.muli %l, %red_slab : index

    %A_access = ktdp.construct_access_tile %A_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>

    %AB_init = tensor.empty() : tensor<128x1x3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile
                              : tensor<128x1x3x64xf16>, tensor<128x1x3x64xf16>)
                          outs(%AB_init : tensor<128x1x3x64xf16>)
                          -> tensor<128x1x3x64xf16>

    %red_init   = tensor.empty() : tensor<128x1x64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16)
                              outs(%red_init : tensor<128x1x64xf16>)
                              -> tensor<128x1x64xf16>
    %partial_3d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<128x1x3x64xf16>)
                    outs(%red_filled : tensor<128x1x64xf16>)
                    dimensions = [2]

    %partial_4d = tensor.expand_shape %partial_3d [[0], [1], [2, 3]]
                    output_shape [128, 1, 1, 64]
                    : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    // Produce: every tile contributes its partial_4d to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles,
        groups                   = #all_groups
        : tensor<128x1x1x64xf16> -> !ktdp.tile_future<tensor<128x1x1x64xf16>>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
    }

    // Multi-group reduce: dim 2 (within-group tile axis) collapsed.
    // Dim 1 (group axis) preserved. Each tile gets its group's <128x1x64>.
    %my_group_result = ktdp.inter_tile_reduce(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        groups                   = #all_groups,
        identity(%add_id : tensor<128x1x1x64xf16>)
        : !ktdp.tile_future<tensor<128x1x1x64xf16>> -> tensor<128x1x64xf16>
    {
      ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
        %init = tensor.empty() : tensor<128x1x1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs
                               : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>)
                           outs(%init : tensor<128x1x1x64xf16>)
                           -> tensor<128x1x1x64xf16>
        ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
    }

    // Post-reduction: each tile writes its group's result to slice [*, g, l, *].
    %my_result_4d = tensor.expand_shape %my_group_result [[0], [1, 2], [3]]
                      output_shape [128, 1, 1, 64]
                      : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x4x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%c0, %g, %l, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x4x64xf16> -> !ktdp.access_tile<128x1x1x64xindex>

    ktdp.store %my_result_4d, %E_access
              : tensor<128x1x1x64xf16>, !ktdp.access_tile<128x1x1x64xindex>

    return
  }
}
```

### 8.3 Reduce-scatter  →  `inter_tile_produce` + `inter_tile_reduce_scatter`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles,
    groups                   = #all_groups
    : tensor<128x1x1x64xf16> -> !ktdp.tile_future<tensor<128x1x1x64xf16>>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
}

// Reduce and scatter; each tile receives its own slice along dim 0.
// scatter_dimension = 0 → 128-row axis split across 4 tiles; each gets <32x1x64>.
%my_chunk = ktdp.inter_tile_reduce_scatter(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    groups                   = #all_groups,
    scatter_dimension        = 0,
    identity(%add_id : tensor<128x1x1x64xf16>)
    : !ktdp.tile_future<tensor<128x1x1x64xf16>> -> tensor<32x1x64xf16>
{
  ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
    %sum = linalg.add ins(%lhs, %rhs ...) ...
    ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
}
// Each tile holds a different slice — ownership explicit via SSA result.
```

#### 8.3.1 Full IR — multi-group reduce-scatter (128×8×12×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x12x64xf16>`
in HBM. The four axes have distinct roles:

- Dim 0 (size 128): the **scatter axis** — within each group, this axis
  is split across that group's 4 tiles.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 12): the **reduction axis** — within each group, 4 tiles
  cooperate over this axis.
- Dim 3 (size 64): vector / stick axis, preserved.

32 tiles, 8 groups of 4. `g = t / 4`, `l = t % 4`. Tile `(g, l)` reads
slice `[*, g, l*3 : l*3+3, *]` — shape `<128x1x3x64>`. The per-tile
pipeline through to `%partial_4d` (shape `<128x1x1x64>`) is identical
to §8.2.2.

The op reduces dim 2 (within-group tile axis, size 1) and scatters dim 0
(128 / 4 = 32 rows per tile). Tile `(g, l)` ends up with rows
`[l*32 : (l+1)*32]` of group `g`'s reduced `<128x1x64>`.

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 2   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

// E view (post-scatter output): 128x8x64.
#E_view_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 63  >= 0)>

// E access tile per writer: 32x1x64 anchored at [l*32, g, 0].
#E_tile_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 31 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 63 >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_reduce_scatter_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %red_slab      = arith.constant 3  : index   // 12 / 4
    %scatter_chunk = arith.constant 32 : index   // 128 / 4

    %A_start = arith.constant 1024     : index
    %B_start = arith.constant 12583936 : index
    %E_start = arith.constant 25166848 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x12x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x12x64xf16>

    // Identity: tensor<128x1x1x64xf16> of zeros — matches partial type T_p.
    %c_zero  = arith.constant 0.0 : f16
    %id_init = tensor.empty() : tensor<128x1x1x64xf16>
    %add_id  = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<128x1x1x64xf16>)
                 -> tensor<128x1x1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %red_anchor = arith.muli %l, %red_slab : index

    %A_access = ktdp.construct_access_tile %A_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>

    %AB_init = tensor.empty() : tensor<128x1x3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile
                              : tensor<128x1x3x64xf16>, tensor<128x1x3x64xf16>)
                          outs(%AB_init : tensor<128x1x3x64xf16>)
                          -> tensor<128x1x3x64xf16>

    %red_init   = tensor.empty() : tensor<128x1x64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16)
                              outs(%red_init : tensor<128x1x64xf16>)
                              -> tensor<128x1x64xf16>
    %partial_3d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<128x1x3x64xf16>)
                    outs(%red_filled : tensor<128x1x64xf16>)
                    dimensions = [2]

    %partial_4d = tensor.expand_shape %partial_3d [[0], [1], [2, 3]]
                    output_shape [128, 1, 1, 64]
                    : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    // Produce: every tile contributes its partial_4d to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles,
        groups                   = #all_groups
        : tensor<128x1x1x64xf16> -> !ktdp.tile_future<tensor<128x1x1x64xf16>>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
    }

    // Reduce dim 2 (within-group tile axis). Scatter dim 0 (chunk = 32).
    // Group axis (dim 1) preserved. Each tile receives <32x1x64>.
    %my_chunk = ktdp.inter_tile_reduce_scatter(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        groups                   = #all_groups,
        scatter_dimension        = 0,
        identity(%add_id : tensor<128x1x1x64xf16>)
        : !ktdp.tile_future<tensor<128x1x1x64xf16>> -> tensor<32x1x64xf16>
    {
      ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
        %init = tensor.empty() : tensor<128x1x1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs
                               : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>)
                           outs(%init : tensor<128x1x1x64xf16>)
                           -> tensor<128x1x1x64xf16>
        ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
    }

    // Post-scatter: tile (g, l) writes rows [l*32 : l*32+32] of group g's result.
    %my_row_anchor = arith.muli %l, %scatter_chunk : index

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 64],
        strides: [512, 64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%my_row_anchor, %g, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_3d
    } : memref<128x8x64xf16> -> !ktdp.access_tile<32x1x64xindex>

    ktdp.store %my_chunk, %E_access
              : tensor<32x1x64xf16>, !ktdp.access_tile<32x1x64xindex>

    return
  }
}
```

---

## 9. Relationship to existing ops

| Existing op | Maps to in this design |
|-------------|------------------------|
| `inter_tile_produce` | `ktdp.inter_tile_produce` — `consumer_tiles_per_group` moved to the delivery op; `producer_tile_per_group` → `producer_tiles_per_group` (generalized to multi-producer) |
| `inter_tile_consume` | `ktdp.inter_tile_consume` — unchanged semantics |
| `inter_tile_reduce` | `ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce` — producer block removed from the reduction op |
| `inter_tile_reduce_scatter` | `ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce_scatter` — producer block removed from the reduction op |

The `!ktdp.tile_future<T>` type is shared across all ops.

The previous `ktdp.inter_tile` single op (Approach B draft) is replaced
by this four-op design: `ktdp.inter_tile` had producer and optional
combiner regions in one op with `consumer_tiles_per_group` determining
delivery mode. The four-op design makes production and delivery explicitly
separate ops, with the delivery mode determined by which delivery op is
chosen rather than by attribute combinations.

---

## 10. Open questions

**Q1. `consumer_tiles` ⊄ `producer_tiles`?**
A tile in `consumer_tiles_per_group` but not in `producer_tiles_per_group`
has no partial to contribute — it would receive the reduced result without
participating in the reduction, implying an implicit identity contribution.
Whether this is allowed and whether identity injection is implicit or must
be explicit in the producer block is still open. The inverse direction
(producer tile outside the consumer set) is explicitly supported: see the
reduce-to-one and reduce-to-subset cases in §4.1.

**Q2. Multi-tensor generalization.**
The existing ops support variadic partials (N ≥ 1 for argmax-style
reductions). The ops here should carry the same variadic structure.
For N = 2 (argmax): two identities, two `ktdp.yield_partial` operands in
the produce block, four combiner arguments yielding two values, two
delivery op results. Each result follows the same per-op type rules
independently.

**Q3. Gather pattern.**
A fourth pattern — all tiles produce, one tile consumes the concatenation
of all partials (no combining) — is not covered. Left as future work.

**Q4. Consume placement.**
Whether the verifier should enforce that delivery ops appear only inside
a guard matching `consumer_tiles_per_group`, or whether this is left to
lowering. If the union of consumer sets equals the set of all executing
tiles, no guard is needed; otherwise a tile not in the consumer set that
reaches the delivery op would be a verifier error.

**Q5. Multiple consumers of a single future.**
Should a single `%future` be usable by more than one delivery op? In SSA
a value may have multiple uses. If allowed, each use represents an
independent extraction. This is sound but may require lowering to handle
duplicate barrier waits. Restricting to exactly one delivery op per future
simplifies lowering at the cost of expressiveness.
