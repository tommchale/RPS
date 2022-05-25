# RPS Game

Automated RPS game in which a user plays against a computer. The program accesses the webcam and uses a Keras model to detect whether the user is showing rock, paper, scossors or nothing.

## Teachable Model

Using Teachable machine created four classes: - Rock - Paper - Scissors - Nothing.

By uploading images representing each class the teachable model, using teachable default settings, was presented data to enable differentiation between those four options.


## Setting up the environment

Tensorflow and opencv-python are required for this game. These were installed in a conda virtual environment.

## Writing the code

The first step was creating the underlying RPS theory, which has been left in for reference.

The must haves for the game are outlined below:

1. Compute a random computer choice.
2. Using keras model, identify user choice from what is presented to the user webcam.
3. Identify a round winner.
4. Identify a game winner.

The nice to haves for the game are outlined below:

1. Instructions and countdown for the user to present an option.
2. Round winner and score totals displayed to webcam video screen.
3. Option to quit at multiple points throughout the game.
4. Ability for user to play again if required.

### Identifying output from the Keras Model

The Keras model, based on what was shown to the webcam would output a predicition in a nested list, e.g. [[1.08345895e-08 2.2457958e-14 1.204985e-010 9.9999987e-01]] with the highest value being the class the teachable machine identified. 

In this game the over a period of 0.5 seconds the index of the highest value in the nested list was added to a prediction list. A mode was then taken of prediction list to give the most common class indentified over the 0.5 seconds.

The idea of this was to give the user a degree of flexibility in presenting their option after the countdown, movement during the prediction, and  error of the Teachable machine.

## Future Improvements

1. Cleaner exit from the game, as currently the next step in the get_user_choice function is taken prior to game exit.
2. Refinement of the Teachable machine model to enable greater accuracy in predictions.




