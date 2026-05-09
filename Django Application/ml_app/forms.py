from django import forms

class VideoUploadForm(forms.Form):

    upload_video_file = forms.FileField(label="Select Video", required=True,widget=forms.FileInput(attrs={"accept": "video/*", "class": "form-control-file"}))
    model_name = forms.ChoiceField(label="Detection Model", required=True)

    def __init__(self, *args, **kwargs):
        model_choices = kwargs.pop("model_choices", [])
        super().__init__(*args, **kwargs)
        self.fields["model_name"].choices = model_choices
        self.fields["model_name"].widget.attrs.update({"class": "form-control upload-select"})
