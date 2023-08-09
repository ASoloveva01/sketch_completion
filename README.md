# Приложение для завершения рисования эскизов
  Данное приложение использует архитектуру сети, основанную на сети SketchGAN. SketchGAN - это новый подход на основе генеративно-состязательной сети (GAN), который совместно завершает и распознает эскиз, повышая производительность обеих задач. 
    
  В данной статье применяется каскадная сеть Encode-Decoder для итеративного завершения входного эскиза и используется вспомогательная задача классификации эскиза для распознавания завершенного эскиза.
![Иллюстрация к проекту](https://github.com/ASoloveva01/sketch_completion/blob/main/corrupted_sketch.png)
![Иллюстрация к проекту](https://github.com/ASoloveva01/sketch_completion/blob/main/completed_sketch.png)
## Как установить и запустить
- Клонируем репозиторий
  ```python
  git clone https://github.com/ASoloveva01/sketch_completion
  cd sketch_completion
  ```
- Устанавливаем зависимости
  ```python
  pip install -r requirements.txt
  ```
- Обучение и валидация модели 
  ```python
  python main.py
  ```
- Запуск приложения
  ```python
  python app.py
  ```
# Как пользоваться
В соответсвующем месте начните рисовать эскиз.  
![Иллюстрация к проекту](https://github.com/ASoloveva01/sketch_completion/blob/main/app.png)  
При нажатии "Дорисовать" холст очистится и внизу появится готовый эскиз(при создании скриншота была использована модель с рандомными параметрами).  
![Иллюстрация к проекту](https://github.com/ASoloveva01/sketch_completion/blob/main/result.png)  
# Датасет 
Изображения для обучения были взяты <a href="http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/">отсюда</a>.
# Источники
- F. Liu, X. Deng, Y. -K. Lai, Y. -J. Liu, C. Ma and H. Wang, "SketchGAN: Joint Sketch Completion and Recognition With Generative Adversarial Network," *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Long Beach, CA, USA, 2019, pp. 5823-5832, doi: 10.1109/CVPR.2019.00598.
- Yu, Q., Yang, Y., Liu, F. et al. Sketch-a-Net: A Deep Neural Network that Beats Humans. *Int J Comput Vis* 122, 411–425 (2017). https://doi.org/10.1007/s11263-016-0932-3

