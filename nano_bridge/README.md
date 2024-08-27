# nano_bridge

This package is very particular to our computer setting, intended to account for the fact that the publication rate of one of the two image topics (RGB and depth) becomes very unreliable when you have one subscriber each (which is common). See this issue for more: https://github.com/personalrobotics/ada_feeding/issues/73

`sender` and `receiver` are generic nodes that can serialize any number of ROS topics of any type into one ROS topic. However, pickle serialization can take ~0.2s, which can slow down the publisher. Hence, `sender_compressed_image` and `receiver_compressed_image` can combine any number of CompressedImage topics into one, but can only do it for those topics (and it doesn't have the overhead of serialization).

A future TODO is to create a generic way to handle cases where the input types are the same and are different, and to create a unified sender and receiver class.