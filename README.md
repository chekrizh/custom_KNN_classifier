## Реализация модели KNN для задач классификации  
_используются numpy и scipy(distance metric)_  
 * [knn.py](https://github.com/chekrizh/custom_KNN_classifier/blob/master/knn.py) содержит класс KNearestNeighbor с типичным для моделей интерфейсом взаимодействия (методы fit() и predict()).  
   Реализацию можно упростить и свести только к ф-ции predict(), но fit() реализован так, чтобы можно было дообучить модель, добавив новых данных. (Проверка на уникальность не происходит! Если она необходима - посмотреть какие данные уже есть в модели можно с помощью аттрибута base_data)
* [test.py](https://github.com/chekrizh/custom_KNN_classifier/blob/master/test.py) - пример работы и сравнение accuracy_score с реализацией в sklearn. При равных условиях показывают равноценную точность
