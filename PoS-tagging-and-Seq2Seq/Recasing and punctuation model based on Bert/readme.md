Pretrained model Link (Size: 1.2 GB) : https://drive.google.com/file/d/1zxbp-vrxno0jDywVmg0jk0WVsU6fmGei/view?usp=sharing    


Tested on Colab GPU, Colab CPU (Linux), Local PC (Windows) 
pip3 install -r requirements.txt 

You may need to upgrade the numpy version on local pc.  
After that run the following: (Assuming the pre trained weight and input text is no the same folder): 

python recasepunc.py predict en.23000 <input.txt> output1.txt  

For data preprocessing follow the Daily Report 08 july 2022. Processed data with separator token added into the dataset folder. 

For training: 
python recasepunc.py train train.x train.y valid.x valid.y en.23000 --lang en


Separator restoration custom trained model link: https://drive.google.com/file/d/1AwMmFmDrPQTwF7nLR9YmNkejaPoAFJZd/view?usp=sharing 


Sample output: 

![Screenshot (107)](https://user-images.githubusercontent.com/43466665/178209388-45b22ee2-8562-4590-b40a-8ffd4c2235ee.png)
