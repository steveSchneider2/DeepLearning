	?я???@?я???@!?я???@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?я???@+?@.??@13???V?;@A???߾??I?n?????*	     ?P@2U
Iterator::Model::ParallelMapV2??ZӼ???!??Dz?r>@)??ZӼ???1??Dz?r>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?0?*??!{?rv?>@)e?X???1?~5&?9@:Preprocessing2F
Iterator::Model?X?? ??!2????E@)??y?):??1h???1?*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v????!kL?*g3@)???_vO~?1??@??&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceǺ???v?!M?*g? @)Ǻ???v?1M?*g? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipvOjM??!?n?Wc"L@)	?^)?p?1?¯?Dz@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF%u?k?!6&???@)F%u?k?16&???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 97.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???	mMX@Q"e??^R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+?@.??@+?@.??@!+?@.??@      ??!       "	3???V?;@3???V?;@!3???V?;@*      ??!       2	???߾?????߾??!???߾??:	?n??????n?????!?n?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???	mMX@y"e??^R@