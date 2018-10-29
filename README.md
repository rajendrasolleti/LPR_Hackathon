Objective : Automatic License plate character recognition from closed circuit camera footage

Camera feed placed at the entrance would be the input to our application.
The application then generates images from this feed.
The images are then processed by applying multiple filters like gray scaling, thresholding, contouring, background checking, ratio checking to identify the possible number plates in this.
Characters are tried to be fetched from the identified number plate. We have created a ML model for this and trained to identify characters & numbers.
Return the identified number.
The number plates identified are saved into the database which could be mapped to vehicle owner details to be used later for toll collection, security purposes etc. 

Instead of Tesseract use custom built model. We tried using XGBoost. Integration pending.
We could see the accuracy improves if we are giving the car image instead of the whole frame. So we could add an additional to identify the car object first and then apply our logic.
