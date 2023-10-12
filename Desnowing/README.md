### Download the Datasets
- SRRS [[gdrive](https://drive.google.com/file/d/11h1cZ0NXx6ev35cl5NKOAL3PCgLlWUl2/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1VXqsamkl12fPsI1Qek97TQ?pwd=vcfg)]
- CSD [[gdrive](https://drive.google.com/file/d/1pns-7uWy-0SamxjA40qOCkkhSu7o7ULb/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1N52Jnx0co9udJeYrbd3blA?pwd=sb4a)]
- Snow100K [[gdrive](https://drive.google.com/file/d/19zJs0cJ6F3G3IlDHLU2BO7nHnCTMNrIS/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1QGd5z9uM6vBKPnD5d7jQmA?pwd=aph4)]

### Training

~~~
python main.py --mode train --data_dir your_path/CSD
~~~

### Evaluation
#### Download the model [here](https://drive.google.com/drive/folders/1HXCwpDbzRL9KLc9XPhUPf2YisS_1wDxo?usp=sharing)
#### Testing
~~~
python main.py --data_dir your_path/CSD
~~~

For training and testing, your directory structure should look like this

`Your path` 
 `├──CSD` 
     `├──train2500`  
          `├──Gt`  
          `└──Snow`  
     `└──test2000`  
          `├──Gt`  
          `└──Snow`  
 `├──SRRS` 
     `├──train2500`  
          `├──Gt`  
          `└──Snow`  
     `└──test2000`  
          `├──Gt`  
          `└──Snow`  
 `└──Snow100K` 
     `├──train2500`  
          `├──Gt`  
          `└──Snow`  
     `└──test2000`  
          `├──Gt`  
          `└──Snow`  