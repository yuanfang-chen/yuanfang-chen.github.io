---
layout: post
title:  "CUTLASS cute for developer"
# date:   2025-11-11 11:18:26 -0800
categories: CUDA
typora-root-url: ..
---

* TOC
{:toc}
## Library Organization

CuTe is a header-only C++ library, so there is no source code that needs building. Library headers are contained within the top level [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) directory, with components of the library grouped by directories that represent their semantics.

| Directory                                                    | Contents                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) | Each header in the top level corresponds to one of the fundamental building blocks of CuTe, such as [`Layout`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp) and [`Tensor`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/tensor.hpp). |
| [`include/cute/container`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container) | Implementations of STL-like objects, such as tuple, array, and aligned array. |
| [`include/cute/numeric`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric) | Fundamental numeric data types that include nonstandard floating-point types, nonstandard integer types, complex numbers, and integer sequence. |
| [`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm) | Implementations of utility algorithms such as copy, fill, and clear that automatically leverage architecture-specific features if available. |
| [`include/cute/arch`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch) | Wrappers for architecture-specific matrix-matrix multiply and copy instructions. |
| [`include/cute/atom`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom) | Meta-information for instructions in `arch` and utilities like partitioning and tiling. |


## include/cute/numeric/arithmetic_tuple.hpp
a lightweight numeric tuple type defined as `template<class... T> struct ArithmeticTuple : public tuple<T...> { ... };`. It wraps a cute::tuple and provides element-wise numeric semantics (constructors, arithmetic operators, printing, and iterator support).

### ScaledBasis 
ScaledBasis is the CuTe representation of a single “basis” element for arithmetic tuples — i.e., a tuple that is zero everywhere except one leaf/mode where it holds a value. It is used to build and manipulate sparse/per-mode arithmetic increments and to index into hierarchical shapes.
// Shortcuts
`E<> := _1`
`E<0> := (_1,_0,_0,...)`
`E<1>  := (_0,_1,_0,...)`
`E<0,0> := ((_1,_0,_0,...),_0,_0,...)`
`E<0,1> := ((_0,_1,_0,...),_0,_0,...)`
`E<1,0> := (_0,(_1,_0,_0,...),_0,...)`
`E<1,1> := (_0,(_0,_1,_0,...),_0,...)`

## include/cute/container/tuple.hpp
行为和`std::tuple`相同，区别是`Tuple`可以运行在主机端和设备端。
cute::tuple is like std::tuple, with differences:

1. It works on both host and device.
2. Its template arguments must be semiregular types.
3. It is always a standard-layout type if all of its template arguments are standard-layout types.
4. It is always an empty type if all of its template arguments are empty types.

Semiregular types are default constructible and copyable.
They include "value types" like int or float,
but do _not_ include references like int& or float&.
(See std::tie for an example of a tuple of references.)

Standard-layout types preserve ABI across host-device boundaries. They are safe to use as device kernel parameters.
The standard-layout requirement prevents a more common EBO-based implemented of cute::tuple.

The cute::tuple is also simplified over the implementations in std::, cuda::std::, and thrust:: by ignoring much of
the conversion SFINAE, special overloading, and avoiding cvref template types.

Over standard-conforming tuple implementations, this appears to accelerate compilation times by over 3x.

## include/cute/algorithm/tuple_algorithms.hpp
### API列表
`t` represents tuple, when there are multiple `t`s, use `ta`, `tb`, `tc`; `f` represents lambda; `x` means scalar
* `apply(t, f)`: unpack. (t, f) => f(t_0,t_1,...,t_n)
* `transform_apply(t, f, g)`: (t, f, g) => g(f(t_0),f(t_1),...)
* `transform_apply(ta, tb, f ,g)`: (ta, tb, f, g) => g(f(ta_0,tb_0),f(ta_1,tb_1),...)
* `transform_apply(ta, tb, tc, f ,g)`: (ta, tb, tc, f, g) => g(f(ta_0,tb_0,tc_0),f(ta_1,tb_1,tc_1),...)
* `for_each(t, f)`: (t, f) => f(t_0),f(t_1),...,f(t_n). 遍历t的每个mode
* `for_each_leaf(t, f)`: 遍历t的每个叶子节点
* `transform(t, f)`: (t, f) => (f(t_0),f(t_1),...,f(t_n)). 遍历t的每个mode
* `transform(ta, tb, f)`: (ta, tb, f) => (f(ta_0,tb_0),f(ta_1,tb_1),...,f(ta_n,tb_n))
* `transform(ta, tb, tc, f)`: (ta, tb, tc, f) => (f(ta_0,tb_0,tc_0),f(ta_1,tb_1,tc_1),...,f(ta_n,tb_n,tc_n))
* `transform_leaf(t, f)`: 遍历t的每个叶子节点
* `transform_leaf(ta, tb, f)`: 遍历t的每个叶子节点
* `find_if(t, f)`:
* `find(t, x)`:
* `any_of(t, f)`: 功能同find_if，但返回值相反
* `all_of(t, f)`:
* `none_of(t, f)`:
* `filter_tuple(t, f)`: (t, f) => <f(t_0),f(t_1),...,f(t_n)>
* `filter_tuple(ta, tb, f)`:
* `filter_tuple(ta, tb, tc, f)`:
* `fold(t, v, f)`: (t, v, f) => f(...f(f(v,t_0),t_1),...,t_n)
* `fold_first(t, f)`
* `front(t)`: Get the first non-tuple element in a hierarchical tuple
* `back(t)`: Get the last non-tuple element in a hierarchical tuple
* `take<B, E>(t)`: Takes the elements in the range [B,E)
* `select<..>(t)`: Select tuple elements with given indices
* `wrap(t)`: Wrap non-tuples into rank-1 tuples or forward
* `unwrap(t)`: Unwrap rank-1 tuples until we're left with a rank>1 tuple or a non-tuple
* `flatten_to_tuple(t)`: Flatten a hierarchical tuple to a tuple of depth one and wrap non-tuples into a rank-1 tuple.
* `flatten(t)`: Flatten a hierarchical tuple to a tuple of depth one and leave non-tuple untouched.
* `unflatten(t, target_profile)`: Unflatten a flat tuple into a hierarchical tuple
* `insert<N>(t, x)`: Insert x into the Nth position of the tuple
* `remove<N>(t)`: Remove the Nth element of the tuple
* `replace(t, x)`: Replace the Nth element of the tuple with x
* `replace_front(t, x)`: Replace the first element of the tuple with x
* `replace_back(t, x)`: Replace the last element of the tuple with x
* `tuple_repeat<N>(x)`: Make a tuple of Xs of tuple_size N
* `repeat<N>(x)`: Make repeated Xs of rank N
* `repeat_like(t, x)`: Make a tuple of Xs the same profile as tuple T
* `group(t)`: Group the elements [B,E) of a T into a single element. e.g. group<2,4>(T<_1,_2,_3,_4,_5,_6>{}) => T<_1,_2,T<_3,_4>,_5,_6>{}
* `append<N>(t, x)`: Extend a T to rank N by appending/prepending an element
* `append(t, x)`: Extend a T to rank N by appending/prepending an element
* `prepend<N>(t, x)`: Extend a T to rank N by appending/prepending an element
* `prepend(t, x)`: Extend a T to rank N by appending/prepending an element
* `iscan(t, v, f)`: Inclusive scan (prefix sum)
* `escan(t, v, f)`: Exclusive scan (prefix sum)
* `zip(t)`: Zip (Transpose)  
  Take       `((a,b,c,...),(x,y,z,...),...)`        rank-R0 x rank-R1 input<br>
  to produce `((a,x,...),(b,y,...),(c,z,...),...)`  rank-R1 x rank-R0 output
* `zip(t0, t1, ..., ts)`: `zip(cute::make_tuple(t0, t1, ts...))`
* `zip2_by(t, guide, seq<Is...>, seq<Js...>)`:
    A guided zip for rank-2 tuples<br>
    Take a tuple like `((A,a),((B,b),(C,c)),d)`<br>
    and produce a tuple `((A,(B,C)),(a,(b,c),d))`<br>
    where the rank-2 modes are selected by the terminals of the guide `(X,(X,X))`

* `reverse(t)`: return A tuple of the elements of `t` in reverse order

## IntTuple

CuTe defines the IntTuple concept as either an integer, or a tuple of IntTuples. Note the recursive definition. 

* IntTuple是一个Concept，可以理解为模版参数，不是一个类，也没有具体的定义
* IntTuple的嵌套定义用图来表示就是一个DAG，这个DAG的形状在cute中被称为`profile`。如果两个IntTuple的`profile`相同，则它们是congruent的。
* terminal... 


### API列表

* `get<I>(IntTuple)`: The `I`th element of the `IntTuple`, with `I < rank`. For single integers, `get<0>` is just that integer.

* `rank(IntTuple)`: The number of elements in an `IntTuple`. A single integer has rank 1, and a tuple has rank `tuple_size`.

* `shape(IntTuple)`: ??

* `max`: DAG所有leaf node的最大值

* `min`: DAG所有leaf node的最小值

* `gcd`: DAG所有leaf node的最大公约数

* `depth`: DAG的深度
* `depth(IntTuple)`: The number of hierarchical `IntTuple`s. A single integer has depth 0, a tuple of integers has depth 1, a tuple that contains a tuple of integers has depth 2, etc.

* `product`:

* `product_each`:

* `product_like`:

* `size(IntTuple)`: The product of all elements of the `IntTuple`.

* `sum`:

* `inner_product`:

* `ceil_div`:

* `round_up`:

* `shape_div`:

* `elem_scale`:

* `congruent`: Test if two IntTuple have the same profile (hierarchical rank division)

* `weakly_congruent`: Test if two IntTuple have the similar profiles up to Shape A (hierarchical rank division), weakly_congruent is a partial order on A and B: A <= B

* `compatible`:
    ```cpp
    /** Test if Shape A is compatible with Shape B:
    *    the size of A and B are the same, and
    *    any coordinate into A can also be used as a coordinate into B
    * Equivalently, the size of Shape B is the same as Shape A at each terminal of Shape A.
    * compatible is a partial order on A and B: A <= B
    */
    ```
* `evenly_divides`:
    ```cpp
    /** Test if Shape A is evenly divided by Tiler B
    * @returns Static or dynamic boolean
    * @post if result is true_type, then
    *       size(a) == logical_divide(make_layout(shape(a)),b) will always compile
    *       and result in true_type.
    */
    ```
* `filter_zeros`:


### Converters and constructors with arrays and params
* `make_int_tuple(t, n, init)`
    ```cpp
    /** Make an IntTuple of rank N from an Indexable array.
    * Access elements up to a dynamic index n, then use init (requires compatible types)
    * Consider cute::take<B,E> if all indexing is known to be valid
    * \code
    *   std::vector<int> a = {6,3,4};
    *   auto tup = make_int_tuple<5>(a, a.size(), 0)            // (6,3,4,0,0)
    * \endcode
    */
    template <int N, class Indexable, class T>
    CUTE_HOST_DEVICE constexpr
    auto
    make_int_tuple(Indexable const& t, int n, T const& init)
    ```
* `fill_int_tuple_from`
    ```cpp
    /** Fill the dynamic values of a Tuple with values from another Tuple
    * \code
    *   auto params = make_tuple(6,3,4);
    *   cute::tuple<Int<1>, cute::tuple<int, int, Int<3>>, int, Int<2>> result;
    *   fill_int_tuple_from(result, params);                    // (_1,(6,3,_3),4,_2)
    * \endcode
    */
    template <class Tuple, class TupleV>
    CUTE_HOST_DEVICE constexpr
    auto
    fill_int_tuple_from(Tuple& result, TupleV const& vals)
    ```
* `make_int_tuple_from`
    ```cpp
    /** Make a "Tuple" by filling in the dynamic values in order from the arguments
    * \code
    *   using result_t = cute::tuple<Int<1>, cute::tuple<int, int, Int<3>>, int, Int<2>>;
    *   auto result = make_int_tuple_from<result_t>(6,3,4);     // (_1,(6,3,_3),4,_2)
    * \endcode
    */
    template <class Tuple, class... Ts>
    CUTE_HOST_DEVICE constexpr
    Tuple
    make_int_tuple_from(Ts const&... ts)
    ```
* `to_array`
    ```cpp
    /** Convert a tuple to a flat homogeneous array of type T
    * \code
    *   auto tup = cute::make_tuple(Int<1>{}, cute::make_tuple(6,3,Int<3>{}),4,Int<2>{});
    *   cute::array<uint64_t,6> result = to_array<uint64_t>(tup);   // [1,6,3,3,4,2]
    * \endcode
    */
    template <class T = int64_t, class IntTuple>
    CUTE_HOST_DEVICE constexpr
    auto
    to_array(IntTuple const& t)
    ```

### Comparison operators
Lexicographical comparison
* `lex_less(IntTupleA, IntTupleB)`:
* `lex_leq(IntTupleA, IntTupleB)`: `!lex_less(IntTupleB, IntTupleA)`
* `lex_gtr(IntTupleA, IntTupleB)`: `lex_less(IntTupleB, IntTupleA)`
* `lex_geq(IntTupleA, IntTupleB)`: `!lex_less(IntTupleA, IntTupleB)`

Colexicographical comparison
* `colex_less(IntTupleA, IntTupleB)`:
* `colex_leq(IntTupleA, IntTupleB)`: `!colex_less(IntTupleB, IntTupleA)`
* `colex_gtr(IntTupleA, IntTupleB)`: `colex_less(IntTupleB, IntTupleA)`
* `colex_geq(IntTupleA, IntTupleB)`: `!colex_less(IntTupleA, IntTupleB)`

Elementwise [all] comparison
* `elem_less(IntTupleA, IntTupleB)`: returns true if every 
* `elem_leq(IntTupleA, IntTupleB)`: `!elem_less(IntTupleB, IntTupleA)`
* `elem_gtr(IntTupleA, IntTupleB)`: `elem_less(IntTupleB, IntTupleA)`
* `elem_geq(IntTupleA, IntTupleB)`: `!elem_less(IntTupleA, IntTupleB)`


## Layout algebra
### Layout construction
* `make_shape(Ts const&... t)`
* `make_stride(Ts const&... t)`
* `make_step(Ts const&... t)`
* `make_coord(Ts const&... t)`
* `make_tile(Ts const&... t)`
* `make_layout(Shape const& shape, Stride const& stride)`
* `make_layout(Shape const& shape)`

### Convenience tags for common layouts
* `make_layout(Shape const& shape, LayoutLeft)`
* `make_layout(Shape const& shape, LayoutRight)`

### Construct a layout from multiple layouts by concatenation
```cpp
make_layout(Layout<Shape0,Stride0> const& layout0);

make_layout(Layout<Shape0,Stride0> const& layout0,
            Layout<Shape1,Stride1> const& layout1);

make_layout(Layout<Shape0,Stride0> const& layout0,
            Layout<Shape1,Stride1> const& layout1,
            Layout<Shapes,Strides> const&... layouts);
```

### Advanced Layout constructions
* `make_ordered_layout`
* `make_layout_like`
* `make_fragment_like`
* `make_fragment_like`
* `make_identity_layout`: Make an identity layout that maps a coordinate to itself

### Operations to manipulate Layouts like a tuple of pairs
* `get(layout)`
* `take`
* `select`
* `flatten(layout)`: Return a layout with depth at most 1
* `unflatten(layout)`: Return a layout whose profile is congruent to TargetProfile

### Utilities
* `layout<...I>()`: Return the sublayout of mode I...
* `shape(layout)`: Return the shape of a mode
* `stride(layout)`: Return the stride of a mode
* `size(layout)`: Return the number of elements in a mode
* `rank(layout)`: Return the number of modes
* `depth(layout)`: Return the depth of the layout
* `coprofile(layout)`: Return the coprofile of a mode as a tuple of _0s
* `coshape(layout)`: Return the codomain shape of a mode
* `cosize(layout)`: Return the codomain size of a mode
* `crd2idx(layout)`: With crd2idx(coord, shape), makes sense to have crd2idx(coord, Layout) as well

### Slice and Dice a layout
```cpp
slice(Coord const& c, Layout<Shape,Stride> const& layout);

slice_and_offset(Coord const& c, Layout<Shape,Stride> const& layout);

dice(Coord const& c, Layout<Shape,Stride> const& layout);

domain_offset(Coord const& coord, Layout<Shape,Stride> const& layout);
```

### Transform the modes of a layout
transform_layout
transform_layout

### Coalesce and Filter
* `coalesce(layout)`: 对Layout的mode从右往左扫描，apply一下逻辑
* coalesce(layout, target_profile)
* coalesce(shape): Combine static and dynamic modes of a shape.
* `filter_zeros(layout)`: Replace the modes in layout that have a 0-stride with a 1-size
* `filter_zeros(layout, target_profile)`: Replace the modes in layout that correspond to a 0 at the terminals of trg_profile with a 1-size
* `filter(layout)`: Remove all of the 0-strides and 1-sizes, Return 1-shape if empty
* `filter(layout, target_profile)`: Apply filter at the terminals of trg_profile

### Append, Prepend, replace
* append
* append
* prepend
* prepend
* replace
* `group<B,E>(layout)`

### Composition of two layouts: lhs o rhs
* `composition(layoutA, layoutB)`
* `composition(layout, tiler)`: `tiler` is tuple, or static integral, or dynamic integral

### Complement
* `complement(layout, cotarget)`
* `complement(layout)`

### Right-Inverse and Left-Inverse
right_inverse(layout)
left_inverse(layout)

### Max Common Layout
* `max_common_layout(layoutA, layoutB)`: Return a layout that points to the maximum number of contiguous elements that logically correspond in the layouts of a and b
* `max_common_vector(layoutA, layoutB)`: Return Int<N> such that N is the maximum number of contiguous elements that logically correspond in the layouts of a and b
* `domain_distribute(ShapeA, ShapeB)`: Return a layout that distributes ShapeB over ShapeA
* `nullspace(layout)`: Kernel (Nullspace) of a Layout

### zip
* `zip(layout)`
* `zip(layoutA, layoutB)`

### Tile unzip
* tile_unzip
```cpp
//
// Tile unzip
//   Logical product and logical divide (on layouts) produce rank-2 results by design.
//   Follow the profile of @a tile and zip the rank-2 modes located at the terminals into
//   their own mode.
//

template <class LShape, class LStride, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
tile_unzip(Layout<LShape,LStride> const& layout,
           Tiler                  const& tiler)
```

### Logical divide
* `logical_divide(layoutA, layoutB)`
* `logical_divide(layoutA, tiler)`
* `ceil_div`

```cpp
// Generalization of ceil_div for Layout lhs
//   is effectively the "rest mode" of logical_divide.
// Occurs in the calculation of gridDim, for example, for generalized tilers
// Example:
//   dim3 gridDim(size(ceil_div(problem_shape_M, cta_tiler_M)),
//                size(ceil_div(problem_shape_N, cta_tiler_N)));
// This does not consider compositional acceptance, so it may be the case that
//   ceil_div produces a result while logical_divide (and friends) do not.
template <class Target, class TShape, class TStride>
CUTE_HOST_DEVICE constexpr
auto
ceil_div(Target                 const& target,
         Layout<TShape,TStride> const& tiler)
```



## ComposedLayout

TODO

## Reference

1. [GPUMode - Lecture 57: CuTe](https://www.youtube.com/watch?v=vzUhbDO_0qk)
1. [知乎 - 深入分析CUTLASS系列](https://zhuanlan.zhihu.com/p/677616101)
1. [知乎 - cute 之 Layout](https://www.zhihu.com/people/reed-84-49/posts)
1. [CUTLASS CUTE笔记](https://www.zhihu.com/people/li-yi-xing-29/posts)

