import random
import cv2
from keras.models import load_model
import numpy as np
import time


class RPS:
    def __init__(self):
        self.rps_options = ['rock', 'paper', 'scissors']

    def get_computer_choice(self):

        self.computer_choice = random.choice(self.rps_options)

        return self.computer_choice

    def get_user_choice(self):
        # loads the tenable model
        model = load_model('keras_model.h5')
        # opens the camera and defines a capture object
        cap = cv2.VideoCapture(0)
        # defines shape of data array
        # an Array of 1 dimensions 224*224*3
        # numpy.ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)[source]
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        timeout = time.time() + 5
        while time.time() < timeout:

            # Captures the video frame by frame
            ret, frame = cap.read()
            # resize takes cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
            # where src is the original image
            # dsize is the desired size
            # interpolation decides which pixel gets which value
            resized_frame = cv2.resize(
                frame, (224, 224), interpolation=cv2.INTER_AREA)
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) /
                                127.0) - 1  # Normalize the image
            data[0] = normalized_image
            prediction = model.predict(data)
            if prediction[0][0] > 0.5:
                print('Rock')
                self.user_choice = "rock"
            elif prediction[0][1] > 0.7:
                print('Scissors')
                self.user_choice = "scissors"
            elif prediction[0][2] > 0.5:
                print("Paper")
                self.user_choice = "paper"
            elif prediction[0][3] > 0.5:
                print("Nothing")

            # shows the image from the frame read by the camera
            cv2.imshow('frame', frame)
            # Press q to close the window
            print(prediction)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        cap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

        return self.user_choice

    def get_winner(self):

        self.get_computer_choice()
        self.get_user_choice()

        if self.computer_choice == self.user_choice:
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')
            print('draw')

        elif (self.computer_choice == 'paper' and self.user_choice == 'rock') or (self.computer_choice == 'scissors' and self.user_choice == 'paper') or (self.computer_choice == 'rock' and self.user_choice == 'scissors'):
            self.winner = 'computer'
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')
            print('\n The winner is...', self.winner)
            return self.winner

        else:
            self.winner = 'user'
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')
            print('\n The winner is...', self.winner)

            return self.winner


def play():
    game = RPS()
    game.get_winner()


play()
