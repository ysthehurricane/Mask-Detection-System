from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField


class Uploadform(FlaskForm):
    picture = FileField('Upload Picture', validators=[FileAllowed(['jpg', 'png'])])
    submit = SubmitField('Upload')