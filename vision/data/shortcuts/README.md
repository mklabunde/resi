### Dataloaders for the Short-Cut experiment;

Being able to distinguish if two models utilize the same shortcut would be a desireable property for similarity metrics.
Here we design a few simple toy-experiments that allow to:
1. Train a model with different shortcuts
2. Test/Verify a model learned the shortcut by omission of the shortcut
3. Test how current similarity metrics can distinguish between two models that utilize the same shortcut.

#### Colored-Dot shortcut:
A colored dot of a chosen color is a simple pattern that should be easy to learn, i.e. a colored dot of fixed size.
Algorithm:
 - Each class gets assigned a unique color
 - During training each image of each class get a colored dot of the class color placed in it (with some probability).
   - If true, a dot of correct color is placed in the image
   - If false, a dot of a random color is placed in the image
   - (The probability can vary between 0, 25, 50, 75, 100%) -- 0 represent no correlation and is there to assure no domain shift in a model.

