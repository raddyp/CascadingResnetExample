Simplified Cascading Resnet architecture example

The Resnet architecture as described in the paper "Deep Residual Learning for Image Recognition" by He et al. has two type of residual blocks in the entire residual layer stack: Identity and Projection.

Residual Blocks:
In this example the projection block is referred to as Convolutional block. The architecture of the Identity and Convolutional block used in this example is shown below.
The residual blocks, both identity and convolutional follow the design as per the 'Resnet50' model with three weights or convolution layers and activation happening after the addition of the skip connection.
![residual_block_types](https://github.com/raddyp/CascadingResnetExample/assets/150963154/efabd015-0e4d-4e5a-94e2-8979f5318eb1)

Residual Layer Stack:
A residual stack is a combination of "1 Convoltuional Block + n number of Identity blocks"(n>=1).
A Residual Layer stacks has m number of residual stacks (m>=1).
A resnet architecture can have a short (shallow) or long (deep) residual layer stack depending on the user requirements.

Original Resnet architecture has a fixed definition of the residual layer stack wherein each stage or stack has one convolutional block ('projection') and predefined 'n' number of identity blocks ('identity').
This example shows how to create a cascade of identity blocks progressing down the residual layer stack.
This example also has the ability to create fixed residual layer stack as explained in figure below.
![residual_stack_types](https://github.com/raddyp/CascadingResnetExample/assets/150963154/8194feb1-14c1-42eb-b089-c01e6a609485)

Fixed residual Layer stack:
As shown in figure above, a 'fixed type stack' has one convolutional block and one identity block per stack. 
Since dimensionality reduction occurs at each convolutional block, the number of residual stacks one can have in the model is dependent on input image size.
The number of residual stacks per model can be varied as per your requirements as well.

Cascading residual layer stack:
In this variation, the number of identity blocks per residual stack increases as the stack number increases in the model.
There is no need to manually add extra identity blocks like in a fixed model making the code concise.
This is just a way of increasing the features in the architecture. The preformance may vary based on data, task, filter numbers, filter sizes etc.
Since dimensionality reduction occurs at each convolutional block, the number of residual stacks one can have in the model is dependent on input image size.

Code Description:
1. Function - model_params
    To define model parameters
2. Function - preprocess_data
    To preprocess and augment data
3. Function - identity_block
    Defines the structure of the identity block
4. Function - convolutional block
    Defines the structure of the convoltuional block
5. Function - Res_layerstack
    Defines the structure of the entire residual layer stack
6. Function create_model
    Defines the entire Resnet model

The code also has facilities to visualize plots for accuracy & loss and a detailed precision, recall estimation snippet derived from Scikitlearn.
Files:
1. resnet_original.py - implementation of the original resnet model as per
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-resnet-from-scratch-with-tensorflow-2-and-keras.md
2. resnet_cascading.py - implementation of the cascading resnet model

   


   
