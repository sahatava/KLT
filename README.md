# KLT
KLT object tracking


1- Find corner points based on the eigenvalues of the Hessian matrix.

2- For each corner compute displacement to next frame using the Lucas-Kanade method

3. Store displacement of each corner, update corner positions.

4. Repeat 2 to 3

![Screenshot](screenshot.png)
