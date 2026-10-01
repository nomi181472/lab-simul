# Fourier-CPPNs for Image Synthesis

Mattie Tesfaldet, Xavier Snelgrove, David Vazquez  
Element AI  
Montréal, Canada

mattie,xavier.snelgrove,dvazquez@elementai.com

## Abstract

*Compositional Pattern Producing Networks (CPPNs) are differentiable networks that independently map  $(x, y)$  pixel coordinates to  $(r, g, b)$  colour values. Recently, CPPNs have been used for creating interesting imagery for creative purposes, e.g. neural art. However their architecture biases generated images to be overly smooth, lacking high-frequency detail. In this work, we extend CPPNs to explicitly model the frequency information for each pixel output, capturing frequencies beyond the DC component. We show that our Fourier-CPPNs (F-CPPNs) provide improved visual detail for image synthesis.*

## 1. Introduction

The fields of computer graphics and computer vision have a long history of introducing new computational approaches for the creation of images. Recently, exciting new deep learning approaches for synthesizing images have come about, such as Generative Adversarial Networks (GANs) [6] for generating realistic faces [10, 11], Convolutional Networks (ConvNets) for image style transfer [5, 17, 8], and Compositional Pattern Producing Networks (CPPNs) [16, 9, 7, 12, 13, 15] for creating aesthetically interesting high-resolution images for creative purposes. The proposed research extends CPPNs by explicitly modelling frequency information. Our experiments show that our Fourier-CPPN (F-CPPN) produces images with improved visual detail while maintaining the advantages of CPPNs.

### 1.1. Compositional Pattern Producing Networks (CPPNs)

CPPNs are differentiable networks that independently map  $(x, y)$  pixel coordinates to  $(r, g, b)$  colour values via the composition and combination of various simple activation functions (Fig. 1 top), e.g. linear, exponential, periodic, etc. Originally, CPPNs were designed for analysing the properties of natural developmental encodings [16] and

The diagram illustrates the architecture of Fourier-CPPNs (F-CPPNs) compared to conventional CPPNs.   
**Conventional CPPN:** A 2D grid of pixel coordinates  $(x, y)$  is processed by a network to produce a 3D volume of color values  $(r, g, b)$ , which is then visualized as a Synthesized RGB Image.   
**Our contribution (F-CPPN):** A 2D grid of pixel coordinates  $(x, y)$  is processed by a network to produce a 3D volume of Fourier Coefficients  $(3H_f W_f)$ . This volume is then transformed using an Inverse Discrete Fourier Transform (IDFT) to produce a Synthesized RGB Image with higher visual detail.

Figure 1: Fourier-CPPNs (F-CPPNs) for image synthesis. (top-left) CPPNs are differentiable networks that map  $(x, y)$  pixel coordinates to  $(r, g, b)$  colour values via linear and non-linear transformations. (bottom) We propose F-CPPNs, an extension of CPPNs which explicitly model the frequency information for each pixel output, capturing frequencies beyond what can be captured by CPPNs. This allows for outputs with increased visual detail.

were optimized using a neural-evolutionary approach that augmented the CPPN’s weights *and* topology. Instead of using the same activation function at each layer, the evolutionary process would find the optimal activation from a list of possible activation functions and “grow” the network, increasing its complexity. Recently however, CPPNs have started to be used for creating interesting imagery for creative purposes [7, 12, 13, 15], e.g., neural art. These recent CPPN implementations have avoided using neural-evolutionary approaches in determining the network’s architecture and have opted for using a fixed architecture instead, whose weights are optimized via gradient descent.

CPPNs have several useful properties. The image they parameterize can be generated at arbitrary resolutions and at arbitrary crops. They can also easily integrate into computer graphics pipelines [15]. However, the locality of pixel coordinates and choice of smooth activation functions constrainsthe resulting image such that colour tends to smoothly vary across neighbouring pixels. As a consequence of this inductive bias, CPPNs create images that appear overly smooth, lacking high-frequency detail that can otherwise add realism to the synthesized image. Up until now, CPPNs have not been designed to incorporate frequency information beyond the DC (zero-frequency) component. We propose an extension to CPPNs, based on Fourier analysis, where each pixel’s colour value is represented as a linear combination of complex-valued sinusoids.

## 1.2. Fourier synthesis

Here we briefly review the relevant theory and mathematics before introducing F-CPPNs. The two-dimensional (2D) discrete Fourier transform (DFT) allows us to represent an image,  $I$ , as a linear combination of complex-valued sinusoidal basis images with varying frequency. The mixing coefficients of these images are given by the complex-valued  $F[\omega_x, \omega_y]$ , which represent the magnitude and phase of the sinusoid with spatial frequency  $\omega_x$  and  $\omega_y$ . The inverse 2D DFT (IDFT) allows us to synthesize  $I$  from its Fourier coefficients,  $F$ . It is defined, per colour channel, as follows,

$$I_c(x, y) = \frac{1}{\sqrt{WH}} \sum_{\omega_x=0}^{W-1} \sum_{\omega_y=0}^{H-1} F_c[\omega_x, \omega_y] e^{i2\pi(\omega_x x/W + \omega_y y/H)}, \quad (1)$$

where  $I_c(x, y)$  is the intensity at pixel  $(x, y)$  for a particular colour channel  $c$ ,  $F_c[\omega_x, \omega_y]$  is the Fourier coefficient for the given spatial frequencies  $\omega_x$  and  $\omega_y$ , and  $W$  and  $H$  are the width and height of the image in pixels, respectively. Note that the number of frequencies (and coefficients) corresponds to the number of pixels in the input image.  $F$  and  $I$  above are complex-valued functions. Since we seek to generate a real-valued image, in this work we simply take the real part of  $I$ .

## 2. Fourier CPPNs

We propose the *Fourier*-CPPN (F-CPPN), an alternate parameterization to CPPNs where each pixel’s  $(r, g, b)$  colour value is obtained from an IDFT on a set of learned Fourier coefficients (see Fig. 1 bottom),

$$(x, y) \xrightarrow{F\text{-CPPN}} (F_r[\omega_x, \omega_y], F_g[\omega_x, \omega_y], F_b[\omega_x, \omega_y]) \xrightarrow{\text{IDFT}} (r, g, b). \quad (2)$$

Consider a  $H \times W \times 2$  grid of pixel coordinates as inputs to a F-CPPN. Recall that a  $H \times W$  image can be defined by a single set of Fourier coefficients of the same dimensions. We design our F-CPPN to output a smaller number of coefficients with dimensions  $H_F \times W_F$ . However, we allow them to vary at each pixel coordinate. This localized and spatially-varying Fourier parameterization of the image is

defined as follows,

$$I_c(x, y) = \frac{1}{\sqrt{W_F H_F}} \sum_{\omega_x=0}^{W_F-1} \sum_{\omega_y=0}^{H_F-1} F_{\text{cyc}}[\omega_x, \omega_y] e^{i2\pi(\omega_x x/W_F + \omega_y y/H_F)}, \quad (3)$$

where  $F_{\text{cyc}}[\omega_x, \omega_y]$  represents the localized frequency information for spatial frequencies  $\omega_x$  and  $\omega_y$  at pixel coordinate  $(x, y)$  for colour channel  $c$ . The final  $(r, g, b)$  colour value is constructed as  $I(x, y) = (I_r(x, y), I_g(x, y), I_b(x, y))$ , from which the final  $H \times W \times 3$  image  $I$  is obtained.

This image representation is overparameterized. Instead of representing the image with  $H \times W$  Fourier coefficients, it is now represented with  $H \times W \times H_F \times W_F$  localized Fourier coefficients. However, this parameterization has some useful properties. First, a region of an image containing a periodic texture with period  $W_F$  and  $H_F$  in  $x$  and  $y$ , respectively, can be represented with a constant set of localized Fourier coefficients at every pixel location in that region. Second, a region of transition from one periodic texture to another can be represented as an interpolation from one set of localized Fourier coefficients to another. By combining the inductive bias of a CPPN, which tends towards constant or smoothly varying outputs, with the property of the localized IDFT, whereby regions with constant coefficients become regions of constant periodic texture, we arrive at our contribution, the F-CPPN. Our F-CPPN is able to explicitly model frequency information beyond the DC component. We can consider CPPNs as a special case of F-CPPNs, where  $W_F = W_H = 1$ , in other words, a F-CPPN that only captures the DC component.

**Architecture design** Our F-CPPN architecture builds on the CPPN implementation from Mordvintsev et al. [13]. Their network consists of eight  $1 \times 1$  convolutional layers, each with 24 filters and each followed by an activation function  $\phi(a) = (\arctan(a)/0.67, \arctan^2(a)/0.67)$ , where  $a$  is the output from a convolution and  $(\cdot, \cdot)$  is a channel-wise concatenation. Note that CPPNs can be implemented as ConvNets strictly using  $1 \times 1$  convolutions. This implies that no information is shared between neighbouring pixels. Our F-CPPN differs from their CPPN in that the final layer does not directly output  $(r, g, b)$  colour values but instead outputs  $(F_{\text{xyr}}[\omega_x, \omega_y], F_{\text{xyg}}[\omega_x, \omega_y], F_{\text{xyb}}[\omega_x, \omega_y])$  Fourier coefficients, which are then fed to an IDFT to produce  $(r, g, b)$  colour values.

## 3. Experiments

Our goal is to improve the visual detail of images synthesized by CPPNs by explicitly modelling frequency information, arriving at F-CPPNs. We qualitatively evaluate theFigure 2: Image reconstruction via Compositional Pattern Producing Networks (CPPNs) and Fourier-CPPNs (F-CPPNs). (left) Target image. (middle) Mordvintsev et al. [13]’s CPPN output. (right) Our F-CPPN’s output. Notice in the zoomed-in sections that our F-CPPN is able to better reconstruct the textural detail of the cat’s fur.

F-CPPN approach of extending a CPPN’s frequency representation beyond the DC component through an ablation study on two image synthesis tasks, image reconstruction and texture synthesis. For each task, we compare the outputs from our F-CPPN and Mordvintsev et al. [13]’s CPPN (the baseline). Additional results can be found in the appendix.

**Training** The objectives used for optimizing the weights of the F-CPPN and the baseline are described in the following sections. We used L-BFGS [1] for optimization. Results were generated using an NVIDIA Tesla P100 GPU and optimization took about an hour for generating a  $224 \times 224$  image. The input pixel coordinates were set to range between  $[\sqrt{3}, -\sqrt{3}]$  with  $(0,0)$  in the centre. The weights for each layer were initialized randomly with zero mean and a variance equal to  $\sqrt{1/C}$ , where  $C$  is equal to the number of input activations. Biases were initialized to zero. The number of spatial frequencies,  $H_F \times W_F$ , was set to  $10 \times 10$ .

### 3.1. Image reconstruction

We show that F-CPPNs have the capacity to synthesize images with greater detail than the baseline with the straightforward task of image reconstruction. Both a F-CPPN and a CPPN are tasked with reconstructing a given image,  $I$ , to produce an output,  $\hat{I}$ . To optimize the weights of both networks, we use the content loss from Gatys et al. [5],

$$\mathcal{L}_{\text{content}} = \frac{1}{L} \sum_l \|\phi_l(I) - \phi_l(\hat{I})\|_2^2, \quad (4)$$

where  $\phi_l(\cdot)$  are the activations of the  $l$ -th layer of VGG-19 [14] when processing input  $(\cdot)$ ,  $L$  is number of layers used, and  $\|\cdot\|_2$  is the L2 norm. In short, the content loss is computed as the mean squared error (MSE) between feature representations of  $I$  and  $\hat{I}$ . The activations used were from layers *conv1\_1*, *pool1*, *pool2*, *pool3*, and *pool4* of VGG-19.

As shown in Fig. 2, the level of detail captured by the F-CPPN is generally greater than the CPPN’s.

### 3.2. Texture synthesis

A visual texture can be loosely defined as a region of an image with stationary feature statistics. Examples of natural textures can include bark, granite, or sand. Texture synthesis is the process of algorithmically generating new image regions that match the stationary feature statistics of a given source texture. Gatys et al. [4] demonstrated impressive results using the learned filters from VGG-19. Textures were modelled in terms of the normalized correlations between activation maps within several layers of the network. Here we synthesize textures with our F-CPPN and the baseline CPPN using Gatys et al. [4]’s texture objective. The task is as follows. Given a target texture, let  $\mathbf{A}^l \in \mathbb{R}^{N_l \times M_l}$  be its row-vectorized activation maps at the  $l$ -th layer of a ConvNet (in this case, VGG-19).  $N_l$  and  $M_l$  denote the number of activation maps and the number of spatial locations, respectively. The normalized correlations between activation maps within a layer are encapsulated by a Gram matrix,  $\mathbf{G}^l \in \mathbb{R}^{N_l \times N_l}$ , whose entries are given by,  $G_{ij}^l = \frac{1}{N_l M_l} \sum_{k=1}^{M_l} A_{ik}^l A_{jk}^l$ .  $A_{ik}^l$  denotes the activation of feature  $i$  at location  $k$  in layer  $l$  on the target texture. Similarly, given a synthesized texture, let  $\hat{\mathbf{A}}^l \in \mathbb{R}^{N_l \times M_l}$  be its row-vectorized activation maps and  $\hat{\mathbf{G}}^l \in \mathbb{R}^{N_l \times N_l}$  be its Gram matrix, whose entries are given by,  $\hat{G}_{ij}^l = \frac{1}{N_l M_l} \sum_{k=1}^{M_l} \hat{A}_{ik}^l \hat{A}_{jk}^l$ . The final objective is defined as the average of the MSE between the Gram matrices of the target texture and that of the synthesized texture,

$$\mathcal{L}_{\text{style}} = \frac{1}{L} \sum_l \|\mathbf{G}^l - \hat{\mathbf{G}}^l\|_F^2, \quad (5)$$

where  $L$  is the number of ConvNet layers used when computing Gram matrices and  $\|\cdot\|_F$  is the Frobenius norm. This texture objective is also known as the style loss [5]. Similarly to Gatys et al. [4], Gram matrices were computed on layers *conv1\_1*, *pool1*, *pool2*, *pool3*, and *pool4* of VGG-19.Figure 3: CPPNs vs. F-CPPNs for texture synthesis. (left) Target texture of pebbles. (middle) Output from Mordvintsev et al. [13]’s CPPN. (right) Output from our F-CPPN. By explicitly modelling frequencies beyond the DC component, our Fourier parameterization provides an improvement in surface detail on the synthesized pebbles.

In the case of Gatys et al. [4], textures were synthesized by directly optimizing their pixel values. Recent approaches use generative ConvNets to synthesize textures [8, 17], parameterizing the output by the ConvNet’s weights. Our implementation follows a similar approach, however, we use a F-CPPN as the generative network. This is a novel application of CPPNs. Results are shown in Fig. 3. By explicitly modelling frequencies beyond the DC component, our F-CPPN provides an improvement in surface detail on the synthesized texture of pebbles. However, we observe a periodic tiling in the top-left region of the output. Whereas regions of constant output would correspond to untextured regions of constant colour in a CPPN (visible in CPPN output in Fig. 3), in a F-CPPN it would correspond to regions of the same coefficients, resulting in a  $H_F \times W_F$ -periodic tiling throughout the region.

## 4. Conclusion

In this paper, we presented an extension to CPPNs, based on Fourier analysis, which we call Fourier-CPPNs (F-CPPNs). F-CPPNs explicitly model the frequency information for each pixel output, capturing high-frequency detail that can not be captured by CPPNs. We applied our F-CPPN to the tasks of image reconstruction and texture synthesis and showed that the resulting images exhibited greater detail than the images synthesized by a CPPN. We observed a limitation common to both F-CPPNs and CPPNs where regions of constant output manifested themselves as regions of periodic tiling of texture in a F-CPPN’s output and untextured regions of constant colour in a CPPN’s output. Regularization methods may alleviate these issues, which we leave as directions for future work. An advantage of F-CPPNs is the direct manipulation of frequencies, allowing for interesting effects such as phase shifting [3] and band-pass filtering. We aim to explore these techniques in future work.

## References

1. [1] R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu. A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*, 16(5):1190–1208, 1995.
2. [2] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi. Describing textures in the wild. 2014.
3. [3] W. T. Freeman, E. H. Adelson, and D. J. Heeger. Motion without movement. 1991.
4. [4] L. A. Gatys, A. S. Ecker, and M. Bethge. Texture synthesis using convolutional neural networks. 2015.
5. [5] L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. 2016.
6. [6] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. 2014.
7. [7] D. Ha. Generating large images from latent vectors, 2016.
8. [8] J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. 2016.
9. [9] A. Karpathy. Image regression, 2014.
10. [10] T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. 2018.
11. [11] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. *arXiv*, 2018.
12. [12] L. Metz and I. Gulrajani. Compositional pattern producing GAN. 2017.
13. [13] A. Mordvintsev, N. Pezzotti, L. Schubert, and C. Olah. Differentiable image parameterizations. *Distill*, 2018.
14. [14] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. *arXiv:1409.1556*, 2014.
15. [15] X. Snelgrove and M. Tesfaldet. Interactive CPPNs in GLSL. 2018.
16. [16] K. O. Stanley. Compositional pattern producing networks: A novel abstraction of development. *GPEV*, 8(2):131–162, 2007.
17. [17] D. Ulyanov, V. Lebedev, A. Vedaldi, and V. S. Lempitsky. Texture networks: Feed-forward synthesis of textures and stylized images. 2016.## A. Additional results for image reconstruction

See Tables 1, 2, 3, and 4 for further image reconstruction results with a CPPN and our F-CPPN on images from the Describable Textures Dataset (DTD) [2]. Recall that the optimized objective is the L2 distance between activations of intermediate layers of a pre-trained VGG-19 network [14].

Note the results for the honeycomb image in the first row of Table 1. The CPPN appears to have insufficient capacity to resolve all of the high-frequency edges of the honeycomb lattice, and so there is a patch that it fails to reconstruct. Our F-CPPN appears to not have the same capacity limitations. We see similar regions where the CPPN fails to reconstruct a portion of the edges of a target image in the second and third rows as well. Note also the “smeared” quality in many of the CPPN images such as the paisley image on the fourth row of Table 1, the scaly image in the second row of Table 4, and the tiger fur image in the third row of Table 4. In all cases, our F-CPPN has a lower MSE (average MSE of  $1.07e5$  compared to CPPN’s average MSE of  $1.6e5$ ), showing that it is not only qualitatively better, but also performs quantitatively better on this particular objective.

## B. Additional results for texture synthesis

Both the CPPN and F-CPPN have more trouble with the texture synthesis objective than with the image reconstruction objective, as shown in Tables 5 and 6. Images from the DTD [2] are used as target textures.

While the CPPN tends to create large regions of constant or smoothly varying colour, the F-CPPN is able to fill these regions with periodic texture containing some of the qualities of the original image. Its results are consistently qualitatively superior than the CPPN’s results. For example, for the bubbles texture in the second row of Table 6, the CPPN captures very few qualities of the original image beyond its approximate colour scheme, whereas the F-CPPN reconstructs both the small and large bubbles, as well as the bubble-like periodic texture.

Although our F-CPPN is not yet achieving the perceptual quality of state-of-the-art texture synthesis algorithms [4, 17], it appears both CPPNs and F-CPPNs are failing to escape local optima. We note that F-CPPNs are able to impressively reconstruct images, down to their higher-frequency textural detail. This leads us to believe that by modifying the texture synthesis objective to take better advantage of the architecture of F-CPPNs, we can improve optimization results. We leave this for future work.

## C. Latent space interpolation

Recall that CPPNs and F-CPPNs accept  $(x, y)$  pixel coordinates as input. Up until this point, we have optimized our F-CPPN on an objective with a single target. We briefly

experimented with optimizing our F-CPPN on multiple targets. This was achieved by modifying the  $(x, y)$  coordinate input by including an additional variable  $\bar{z}$  that conditions the output of the F-CPPN.  $\bar{z}$  is constant for all  $(x, y)$  inputs. We optimized our F-CPPN with the texture synthesis objective against two target textures. The two targets were an image of pebbles and an image of peppers, with  $\bar{z} = (1, 0)$  and  $\bar{z} = (0, 1)$ , respectively.

Figure 4 shows our conditioned F-CPPN’s output during inference time, where  $\bar{z}$  is interpolated for each frame with the function  $\bar{z} = (\cos \theta, \sin \theta)$  and  $\theta$  is incremented from 0 to  $\pi/2$ . The result is an aesthetically pleasing video of a texture of pebbles warping to a texture of peppers.

## D. High-resolution synthesis

CPPNs and F-CPPNs are continuous functions that map  $(x, y)$  pixel coordinates to  $(r, g, b)$  colours; they are not inherently tied to the pixel grid. By feeding sub-pixel coordinates, these continuous functions naturally interpolate pixel colours. This means that high-resolution zoomed-in images can be generated by simply feeding in a more densely sampled coordinate grid as the input. An example of this is shown in Fig. 5 and 6, for both the image reconstruction and texture synthesis objective, respectively.

Note however, that the highest spatial frequency the F-CPPN can output is a hyperparameter of the network architecture itself, so no higher-frequency details other than those visible at the original scale are synthesized as the image is super-sampled. Eventually, the image will appear smooth as the zoom level is increased. This is analogous to the “infinite-resolution” of vector graphics, where the image does not become “pixelated” as the zoom level increases, but no new details are introduced.<table border="1">
<thead>
<tr>
<th>Target image</th>
<th>CPPN</th>
<th>F-CPPN (ours)</th>
</tr>
</thead>
<tbody>
<tr>
<td></td>
<td><br/>1.71e4</td>
<td><br/><b>0.81e4</b></td>
</tr>
<tr>
<td></td>
<td><br/>5.77e4</td>
<td><br/><b>3.25e4</b></td>
</tr>
<tr>
<td></td>
<td><br/>1.48e5</td>
<td><br/><b>4.98e4</b></td>
</tr>
<tr>
<td></td>
<td><br/>1.84e5</td>
<td><br/><b>1.47e5</b></td>
</tr>
</tbody>
</table>

Table 1: Image reconstruction via Compositional Pattern Producing Networks (CPPNs) and Fourier-CPPNs (F-CPPNs). (left) Target image. (middle) CPPN output using the same CPPN architecture as Mordvintsev et al. [13]. (right) Our F-CPPN’s output. Numbers beneath images indicate image reconstruction error (*i.e.*, MSE).<table border="1">
<thead>
<tr>
<th>Target image</th>
<th>CPPN</th>
<th>F-CPPN (ours)</th>
</tr>
</thead>
<tbody>
<tr>
<td></td>
<td><br/>7.28e4</td>
<td><br/><b>5.83e4</b></td>
</tr>
<tr>
<td></td>
<td><br/>3.31e4</td>
<td><br/><b>2.79e4</b></td>
</tr>
<tr>
<td></td>
<td><br/>3.8e4</td>
<td><br/><b>2.26e4</b></td>
</tr>
<tr>
<td></td>
<td><br/>9.83e4</td>
<td><br/><b>8.76e4</b></td>
</tr>
</tbody>
</table>

Table 2: Image reconstruction via Compositional Pattern Producing Networks (CPPNs) and Fourier-CPPNs (F-CPPNs). (left) Target image. (middle) CPPN output using the same CPPN architecture as Mordvintsev et al. [13]. (right) Our F-CPPN’s output. Numbers beneath images indicate image reconstruction error (*i.e.*, MSE).<table border="1">
<thead>
<tr>
<th>Target image</th>
<th>CPPN</th>
<th>F-CPPN (ours)</th>
</tr>
</thead>
<tbody>
<tr>
<td></td>
<td><br/>1.99e5</td>
<td><br/><b>1.52e5</b></td>
</tr>
<tr>
<td></td>
<td><br/>4.31e5</td>
<td><br/><b>3.39e5</b></td>
</tr>
<tr>
<td></td>
<td><br/>3.19e4</td>
<td><br/><b>2.24e4</b></td>
</tr>
<tr>
<td></td>
<td><br/>7.55e4</td>
<td><br/><b>5.76e4</b></td>
</tr>
</tbody>
</table>

Table 3: Image reconstruction via Compositional Pattern Producing Networks (CPPNs) and Fourier-CPPNs (F-CPPNs). (left) Target image. (middle) CPPN output using the same CPPN architecture as Mordvintsev et al. [13]. (right) Our F-CPPN’s output. Numbers beneath images indicate image reconstruction error (*i.e.*, MSE).<table border="1">
<thead>
<tr>
<th>Target image</th>
<th>CPPN</th>
<th>F-CPPN (ours)</th>
</tr>
</thead>
<tbody>
<tr>
<td></td>
<td><br/>7.41e5</td>
<td><br/><b>4.8e5</b></td>
</tr>
<tr>
<td></td>
<td><br/>2.1e5</td>
<td><br/><b>8.16e4</b></td>
</tr>
<tr>
<td></td>
<td><br/>3.36e4</td>
<td><br/><b>1.82e4</b></td>
</tr>
<tr>
<td></td>
<td><br/>1.89e5</td>
<td><br/><b>1.28e5</b></td>
</tr>
</tbody>
</table>

Table 4: Image reconstruction via Compositional Pattern Producing Networks (CPPNs) and Fourier-CPPNs (F-CPPNs). (left) Target image. (middle) CPPN output using the same CPPN architecture as Mordvintsev et al. [13]. (right) Our F-CPPN’s output. Numbers beneath images indicate image reconstruction error (*i.e.*, MSE).<table border="1">
<thead>
<tr>
<th data-bbox="178 91 383 111">Target texture</th>
<th data-bbox="383 91 588 111">CPPN</th>
<th data-bbox="588 91 793 111">F-CPPN (ours)</th>
</tr>
</thead>
<tbody>
<tr>
<td data-bbox="178 111 383 291"></td>
<td data-bbox="383 111 588 291"></td>
<td data-bbox="588 111 793 291"></td>
</tr>
<tr>
<td data-bbox="178 291 383 471"></td>
<td data-bbox="383 291 588 471"></td>
<td data-bbox="588 291 793 471"></td>
</tr>
<tr>
<td data-bbox="178 471 383 651"></td>
<td data-bbox="383 471 588 651"></td>
<td data-bbox="588 471 793 651"></td>
</tr>
<tr>
<td data-bbox="178 651 383 823"></td>
<td data-bbox="383 651 588 823"></td>
<td data-bbox="588 651 793 823"></td>
</tr>
</tbody>
</table>

Table 5: Texture synthesis via Compositional Pattern Producing Networks (CPPNs) and Fourier-CPPNs (F-CPPNs). (left) Target texture. (middle) CPPN output using the same CPPN architecture as Mordvintsev et al. [13]. (right) Our F-CPPN’s output.<table border="1">
<thead>
<tr>
<th data-bbox="179 92 383 111">Target texture</th>
<th data-bbox="383 92 587 111">CPPN</th>
<th data-bbox="587 92 791 111">F-CPPN (ours)</th>
</tr>
</thead>
<tbody>
<tr>
<td data-bbox="179 111 383 291">
</td>
<td data-bbox="383 111 587 291">
</td>
<td data-bbox="587 111 791 291">
</td>
</tr>
<tr>
<td data-bbox="179 291 383 471">
</td>
<td data-bbox="383 291 587 471">
</td>
<td data-bbox="587 291 791 471">
</td>
</tr>
<tr>
<td data-bbox="179 471 383 651">
</td>
<td data-bbox="383 471 587 651">
</td>
<td data-bbox="587 471 791 651">
</td>
</tr>
<tr>
<td data-bbox="179 651 383 823">
</td>
<td data-bbox="383 651 587 823">
</td>
<td data-bbox="587 651 791 823">
</td>
</tr>
</tbody>
</table>

Table 6: Texture synthesis via Compositional Pattern Producing Networks (CPPNs) and Fourier-CPPNs (F-CPPNs). (left) Target texture. (middle) CPPN output using the same CPPN architecture as Mordvintsev et al. [13]. (right) Our F-CPPN’s output.Figure 4: F-CPPN latent space interpolation. Instead of a  $(x, y)$  pixel coordinate input, the F-CPPN accepts a  $(x, y, \vec{z})$  input where  $\vec{z}$  is a variable that conditions the output of the F-CPPN based on a specific target. The two targets in this case are an image of pebbles ( $\vec{z} = (1, 0)$ ) and an image of peppers ( $\vec{z} = (0, 1)$ ). (left-to-right) conditioned output as  $\vec{z}$  is interpolated with the function  $\vec{z} = (\cos \theta, \sin \theta)$  and  $\theta$  is incremented from 0 to  $\pi/2$ . The F-CPPN was optimized with the texture synthesis objective.

Figure 5: High-resolution image reconstruction using F-CPPNs. (left) A  $256 \times 256$  image synthesized by a F-CPPN. (right) A  $1000 \times 1000$  image synthesized by a F-CPPN. CPPNs and F-CPPNs have the benefit of being able to directly synthesize images at any resolution, including resolutions other than the ones they were optimized at. The F-CPPN was optimized with the image reconstruction objective and the resolution of the target was  $256 \times 256$ .Figure 6: High-resolution texture synthesis using F-CPPNs. (left) A  $256 \times 256$  texture synthesized by a F-CPPN. (right) A  $1000 \times 1000$  texture synthesized by a F-CPPN. CPPNs and F-CPPNs have the benefit of being able to directly synthesize images at any resolution, including resolutions other than the ones they were optimized at. The F-CPPN was optimized with the texture synthesis objective and the resolution of the target was  $256 \times 256$ .

