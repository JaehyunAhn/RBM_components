## Restricted Boltzmann Machine Training Set

### Library
* sklearn (Training)
  * BernulliRBM
  * LogisticRegression
  * Pipeline
* sklearn (Data Import)
  * fetch_mldata
* numpy
* pyplot


### Function
#### plot_gallery(title, images, n_col, n_row)
이미지 매트릭스를 불러오고 py-plot을 이용하여 이미지를 불러오는 함수

#### BernulliRBM.components_
히든 유닛 매트릭스(Matrix of Hidden Unit, 즉 Filter)를 가져와서 보여주는 함수 어떤 히든 유닛이 형성되었는지 볼 수 있다.