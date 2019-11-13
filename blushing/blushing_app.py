import cv2
from PIL import Image
import numpy as np
import face_recognition
import scipy.interpolate
from datetime import datetime
from flask import Flask, render_template,request,url_for, jsonify,send_file,send_from_directory
from flask_cors import CORS
app = Flask(__name__,template_folder='templates')
#app=Flask(__name__)
CORS(app)

@app.route("/",methods=['POST','GET'])
def main():
	return render_template('blush_index.html')

@app.route('/blush',methods=['GET','POST'])
def blush():
	if not (request.files['face'].mimetype[:5]=='image'):
		return jsonify(error="Inputs must be face image!")

	face_img = request.files['face']
	#b,g,r,a=request.form.getlist('BGRA[]')
	#face_img.save('input_images\\face_img.jpg')
	ts=datetime.timestamp(datetime.now())
	t,s=str(ts).split('.')
	face_img.save("blush_input_images\\'input_'+{}+'_'+{}+'.jpg'".format(t,s))
	B=request.form.get('B',type=int)
	G=request.form.get('G',type=int)
	R=request.form.get('R',type=int)
	A=request.form.get('A',type=int)
	A=0.1*A
	image=cv2.imread("blush_input_images\\'input_'+{}+'_'+{}+'.jpg'".format(t,s))
	height,width = image.shape[:2]
	imOrg = image.copy()
	face_landmarks_list=face_recognition.face_landmarks(image)    
	point_42=face_landmarks_list[0]['left_eye'][5]
	x_42,y_42=point_42[0],point_42[1]
	point_4=face_landmarks_list[0]['chin'][3]
	x_4,y_4=point_4[0],point_4[1]
	point_32=face_landmarks_list[0]['nose_tip'][0]
	x_32,y_32=point_32[0],point_32[1]
	left_x_b=int((x_42 + 2*x_4 + x_32)/4)
	left_y_b=int((y_42 + 2*y_4 + y_32)/4)
	r=int(0.2*pow((pow(x_4-x_42,2)+pow(y_4-y_42,2)),0.5))

	point_28=face_landmarks_list[0]['nose_bridge'][0]
	point_29=face_landmarks_list[0]['nose_bridge'][1]
	gauss_var=abs(point_28[1]-point_29[1])
	while gauss_var>r:
	    gauss_var=gauss_var/2
	    
	gauss_var=int(gauss_var)
	if gauss_var%2==0:
	    gauss_var+=1


	point_47=face_landmarks_list[0]['right_eye'][4]
	x_47,y_47=point_47[0],point_47[1]
	point_14=face_landmarks_list[0]['chin'][13]
	x_14,y_14=point_14[0],point_14[1]
	point_36=face_landmarks_list[0]['nose_tip'][4]
	x_36,y_36=point_36[0],point_36[1]
	right_x_b=int((x_47 + 2*x_14 + x_36)/4)
	right_y_b=int((y_47 + 2*y_14 + y_36)/4)
	    
	overlay=image.copy()
	cv2.circle(overlay, (left_x_b,left_y_b),r,(B, G, R), -1)
	cv2.circle(overlay,(right_x_b,right_y_b),r,(B, G, R), -1)
	overlay=cv2.GaussianBlur(overlay,(gauss_var,gauss_var),cv2.BORDER_ISOLATED)
	new_image = cv2.addWeighted(overlay, A, image, 1 - A, 0)
	ts1=datetime.timestamp(datetime.now())
	t1,s1=str(ts).split('.')

	cv2.imwrite("blush_output_images\\'blushed_'+{}+'_'+{}+'.png'".format(t1,s1),new_image)
	return send_file("blush_output_images\\'blushed_'+{}+'_'+{}+'.png'".format(t1,s1),mimetype='image/png')

if __name__=='__main__':
	app.run(host='0.0.0.0',debug=True,port=8088)
