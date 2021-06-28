from django import forms


class CombinerForm(forms.Form):
    content_image = forms.ImageField()
    style_image = forms.ImageField()
