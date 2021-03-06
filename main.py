from flask import Flask
from app import views

app = Flask(__name__)

#create rule for url 
# app.add_url_rule('/','index',views.index)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/base','base',views.base)
app.add_url_rule('/cnn','cnn',views.cnn,methods=['GET','POST'])

app.add_url_rule('/vgg16_tf','vgg16_tf',views.vgg16_tf,methods=['GET','POST'])


if __name__ == "__main__":
    # app.run()    
    app.run(debug=True)    