# images_to_grids.py Nodes
InvokeAI nodes for `Images To Grids`, `XYImage To Grid` and supporting nodes

Known Bugs:
- The Grids are added to the gallery it will not update. To see the grid(s) in the gallery you will need to refresh your browser.
- In the `XYImage To Grid` The default colour for the font is white but the node shows this as black. 

## Two Main Nodes  

### 1. "Images To Grids" node - Takes a collection of images and creates a grid(s) of images. If there are more images than the size of a single grid then mutilple grids will be created until it runs out of images.
  - Images: This is the collection of images to add to the grids. It is used by collecting the images with a `Collect` node and feeding this into this input
  - Columns: The number of images across in the grids
  - Rows: The maximum rows per grid
  - Space: The number of pixels gap between images
  - Scale Factor: How much to resize the images by (values of 0...1 will reduce the image size, This is recomended if you have large images or large grids)
  - Resample Mode: Resize type to use
  - Background Color: Background color of the grid

### 2. "XYImage To Grid" node - Converts a collection of XYImages into a labeled Grid of images.  The XYImages collection has to be built using the supporoting nodes. See example node setups for more details.
  - Xyimages: This is the collection of images and XY items to add to the grid. It is used by `XYImage Collect` node and a `Collect` node and feeding this into this input
  - Space: The number of pixels gap between images
  - Scale Factor: How much to resize the images by (values of 0...1 will reduce the image size, This is recomended if you have large images or large grids)
  - Resample Mode: Resize type to use
  - Background Color: background color of the grid
  - Label Font Name: Font to use for the labels Default "Ariel.ttf"
  - Label Front Size: Size of the font to use for lables Default 35
  - Top Label Height: Vertical size to the lables space at the top of the grid.
  - Left Label Width: Horizontal Size of the labels space at the left of the grid.

## Supporting Nodes:
1. Floats To Strings: Takes a float or collections of floats and converts it to a collection of string. Output of this is intended for the `XY Collect` node.
2. Ints To Strings: Takes an int or collection of ints and converts it to a collection of string. Output of this is intended for the `XY Collect` node.
3. XY Collect:  This takes two string collections and converts it into another string collection that is the Product of the first two collections (every combination of the input collections). This is ment to be fed into the `Iterate` node so you can do a generation for each xy pair.
4. XY Expand: This takes the output of the iterate node and expands it to individual x and y items as strings. The output of this needs to passed into the `XYImage Collect` Node as is. However before been used in standard nodes they will need to be converted into the correct type. This can be done with the `String To Int` and `String To Float` Nodes. 
5. String To Float: Converts a string to a float this is needed to convert the output of the `XY Expand` node before it can be used by standard nodes.
6. String To Int: Converts a string to an int this is needed to convert the output of the `XY Expand` node before it can be used by standard nodes.
7. XYImage Collect: The job of this node is to collect the Image and the X and Y items in a single place so they can be passed onto the `Collect` node then onto the `XYImage To Grid` node.
8. CSV To Strings: Converts a CSV string into a collection of strings
9. XY CSV to Strings: takes two CSV strings and outputs a collection that every combination of X and Y this can be used insted of passing ranges into the the XY collect node

## Using "Image To Grids" node
The `Images To Grids` nodes is fairly stright forward to you you just collect images from multiple generation using a `Collect` and pass it into the nodes with the setting of what you want
### Example node setup for Images To nodes
![image](https://github.com/skunkworxdark/XYGrid_nodes/assets/21961335/1f90d2e9-4a07-471b-b12c-ad4ec852dae2)
### Example images to Grid output 
![image](https://github.com/skunkworxdark/XYGrid_nodes/assets/21961335/5c244862-dbcf-4c6f-b021-059bc4f66f10)
![image](https://github.com/skunkworxdark/XYGrid_nodes/assets/21961335/1b3c8ec8-bc06-4dde-bacb-0a81b067b97b)

## Using "XYImages To Grid" node
The `XYImage To Grid` Node is much more involved as you need to setup and preserve the order of the X and Y data to allow the grid to be rendered in a way that makes any sense.
### Eample node layout for XYImage To Grid
![image](https://github.com/skunkworxdark/XYGrid_nodes/assets/21961335/08b1b1e0-de87-492e-941d-607b32bd2e7c)
The lefthand side creates the ranges and combines then into every combination and then passes them into the iterate and then passes them out on the image ganeration part of the node.  Due to the current node limitations (unable to accept int and float into a single input) you have to convert the int or float range inputs into to strings so you have string types passed around then they have to be converted back to the correct type to use them in the text to latents node etc. Ideally I would like to fold it all into a single xy iterate node with  X, Y input that accept collections of any type and output the x,y on the other side.   Any help or suggestion on this would be good.
![image](https://github.com/skunkworxdark/XYGrid_nodes/assets/21961335/7d2c1d3c-5207-4c96-a313-e59311cfc864)
The `XY Expand` node also passes the unconverted X,Y to the `XYImage Collect` Node where the X,Y & image_name are put into a string collection (again it seems as I am unable to work out how to pass a collection of multiple types around). This is then passed into the `Collect` node when the `XYImage To Grid` node can then reconstruct the correct order for the grid and build the grid image.
![image](https://github.com/skunkworxdark/XYGrid_nodes/assets/21961335/c2e8603c-24b1-47a3-a300-9b864854d47e)
### Example XY Grid output
![image](https://github.com/skunkworxdark/XYGrid_nodes/assets/21961335/4711596a-d117-4b11-a39f-887b2e171cca)

![image](https://github.com/skunkworxdark/XYGrid_nodes/assets/21961335/885a8720-0769-48b6-b5ee-09f7f2acb421)



### Images to Grids Node
![image](https://github.com/skunkworxdark/XYGrid_nodes/assets/21961335/04b99d55-a2cd-4b49-940a-4ae4f1ccfc55)
### XYImage to Grid Node
![Screenshot 2023-07-31 171623](https://github.com/skunkworxdark/XYGrid_nodes/assets/21961335/442761a9-9ed4-48b6-9d93-1c277f428395)


