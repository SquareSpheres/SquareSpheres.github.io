# Cuda Algorithms

## Connected Components

### ShiloachVishkin Variant

<p>A CUDA implementation of algorithm 1 and 2, published by Guojing Cong and Paul Muzio</p>

<p>Both algorithms works by initially setting each vertex as its own component, and then connecting higher numbered component to lower components. This creates a star for each component where the lowest numbered vertex in the component is in the center. Algorithm one and two differ on by a additional function for the latter.</p>

```
```cuda
__global__ void GraftKernel(std::pair<int, int\> *graph, const int numEdges, int *component)
...
	if ((fromComponent < toComponent) && (toComponent == component\[toComponent\]))
	{
		has_grafted_d = true;
		component[toComponent] = fromComponent;
	}
...
{
```
```


### Stages
<p>A CUDA implementation of algorithm 3 published by Guojing Cong and Paul Muzio</p>

#### Papers
[1] <a href="https://doi.org/10.1007/978-3-319-14325-5_14">Cong G., Muzio P. (2014) Fast Parallel Connected Components Algorithms on GPUs. In: Lopes L. et al. (eds) Euro-Par 2014: Parallel Processing Workshops. Euro-Par 2014. Lecture Notes in Computer Science, vol 8805. Springer, Cham</a></p>

