# Подсчет количества транспорта на дороге

Реализация:

* YOLOv4 (Tiny) + SORT

Состав проекта:

* traffic_count.py - модуль с функциями подсчета транспорта.
* sort.py - модуль с реализацией алгоритма *[SORT](https://github.com/abewley/sort)*
* yolo - каталог с конфигурацией YOLOv4 (Tiny)

Использование:

```
usage: traffic_count.py [-h] [--path PATH] [--save_video] [--show_gui]

optional arguments:
  -h, --help    show this help message and exit
  --path PATH   path to video file
  --save_video  save video with counts
  --show_gui    show opencv GUI

Example: python -m traffic_count --save_video --path <source video file>
```

В текущем каталоге сохраняется файл "output.avi".

Результаты:

* присутствуют ложные срабатывания для классов автобуса и грузовика
