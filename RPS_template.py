import random
import cv2
from keras.models import load_model
import numpy as np


class RPS:
    def __init__(self):
        self.rps_options = ['rock', 'paper', 'scissors', 'nothing']

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

        while True:

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

        self.user_choice = prediction

        return self.user_choice

    def get_winner(self):

        # self.get_computer_choice()
        # self.get_user_choice()

        if self.computer_choice == self.user_choice:
            print('draw')

        elif (self.computer_choice == 'paper' and self.user_choice == 'rock') or (self.computer_choice == 'scissors' and self.user_choice == 'paper') or (self.computer_choice == 'rock' and self.user_choice == 'scissors'):
            self.winner = 'computer'
            print(self.winner)
            return self.winner

        else:
            self.winner = 'user'
            print(self.winner)
            return self.winner


def play():
    game = RPS()
    game.get_user_choice()


play()
