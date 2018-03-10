# Cuda Algorithms

## Connected Components

### ShiloachVishkin Variant

<p>A CUDA implementation of algorithm 1 and 2, published by Guojing Cong and Paul Muzio</p>


<p>Both algorithms works by initially setting each vertex as its own component, and then going through each edge connecting higher numbered component to lower components. This creates a star for each component where the lowest numbered vertex in the component is in the center. This is done through a method named graft.

After grafting is complete, the CPU and GPU synchronize(because we only use the default stream the kernel calls will be queued and we do not need to explicitly synchronize. In the code we have used explicit synchronization to check for errors after each kernel call). We then call a function called shortcut. Another suitable name would be compress. This function simply compresses each component to the root component. E.g. image the graph 4-->2-->1.  The component array would look like this [1][1][x][2]. 1 is the root in this component, and we want every member of this component to be in in the array. This is what the shortcut function does. After the shortcut function is called the component array would be [1][1][x][1].

```cuda
__global__ void GraftKernel(std::pair<int, int> *graph, const int numEdges, int *component)
	{

		const auto tid = blockIdx.x*blockDim.x + threadIdx.x;
		const auto numThreads = gridDim.x * blockDim.x;

		has_grafted_d = false;

		for (auto i = tid; i < numEdges; i += numThreads)
		{

			int fromVertex = graph[i].first;
			int toVertex = graph[i].second;

			int fromComponent = component[fromVertex];
			int toComponent = component[toVertex];

			if ((fromComponent < toComponent) && (toComponent == component[toComponent]))
			{
				has_grafted_d = true;
				component[toComponent] = fromComponent;

			}
	...
	...
```

```cuda
__global__ void ShortcutKernel(int *component, const int numVertices)
	{

		const auto tid = blockIdx.x*blockDim.x + threadIdx.x;
		const auto numThreads = gridDim.x * blockDim.x;

		for (auto i = tid; i < numVertices; i += numThreads)
		{
			while (component[i] != component[component[i]])
			{
				component[i] = component[component[i]];
			}
		}
	}
```


The only difference between algorithm one and two is that algorithm two has a function that updates the graph itself instead of just updating the component array. This makes memory access more coalesced


```cuda
__global__ void UpdateKernel(std::pair<int, int> *graph, const int numEdges, int *component)
	{
		const auto tid = blockIdx.x*blockDim.x + threadIdx.x;
		const auto numThreads = gridDim.x * blockDim.x;

		for (auto i = tid; i < numEdges; i += numThreads)
		{
			graph[i].first = component[graph[i].first];
			graph[i].second = component[graph[i].second];
		}
	}
```

</p>






### Stages
<p>A CUDA implementation of algorithm 3 published by Guojing Cong and Paul Muzio</p>

#### Papers
[1] <a href="https://doi.org/10.1007/978-3-319-14325-5_14">Cong G., Muzio P. (2014) Fast Parallel Connected Components Algorithms on GPUs. In: Lopes L. et al. (eds) Euro-Par 2014: Parallel Processing Workshops. Euro-Par 2014. Lecture Notes in Computer Science, vol 8805. Springer, Cham</a></p>

