from flask import Flask,render_template,request,send_file,url_for
from flask_cors import cross_origin
from utills.utils import decode
from detector import *
import pandas as pd
import cv2

def frame():
    df = pd.read_csv('result.csv')
    del df['Unnamed: 0']
    df = df.reset_index(drop=True)
    return df


app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/result',methods=['GET','POST'])
@cross_origin()
def result():
    globals()
    if request.method=='POST':
        image = request.json['image']
        img = decode(image).copy()
        ###################################
        tx = start(img)
        if len(rois) == 0:
            img1 = cv2.imread('output/object.jpg')
            txt = 'Put Car Image'
            cv2.rectangle(img1, (214, 0), (400, 24), (255, 50, 0), cv2.FILLED)
            cv2.putText(img1, txt, (216, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Detection', img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if len(rois) > 0:
            if_damage(rois)
            if len(damage) == 0:
                txt = 'found no damage..'
                img2 = cv2.imread('output/car.jpg')
                cv2.imshow('Detection', img2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                #if damage
                txt = 'Found damage'
                ####################
                img3 = cv2.imread('output/damage.jpg')
                cv2.imshow('Detection', img3)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ###################
                # detect position
                position(rois)
                # detect damage condition
                Damage()
                # process data
                results()
                txt2 = pos[0]
                txt3 = con[0]
                txt4 = amount[0]
                ################################
                img4 = cv2.imread('output/position.jpg')
                cv2.imshow('Detection', img4)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ####################
                img5 = cv2.imread('output/damage_cond.jpg')
                cv2.imshow('Detection', img5)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                df = frame()
                make_results(df)
                del img
                ###################
                img7 = cv2.imread('bg.png')
                cv2.putText(img7, 'Damage   : yes', (40, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (163,126,126), 2)
                cv2.putText(img7, f'Position   : {txt2}', (40, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (163,126,126), 2)
                cv2.putText(img7, f'Condition : {txt3}', (40, 140), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (163,126,126), 2)
                cv2.putText(img7, f'Amount   : {txt4}', (40, 190), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (163,126,126), 2)
                cv2.imshow('Results', img7)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print(txt)
                return render_template('home.html')
                ####################################
    else:
        pass
    return render_template('home.html')

@app.route('/download')
def download_file():
    p = 'result.csv'
    return send_file(p,as_attachment=True)


if __name__=="__main__":
    app.run()