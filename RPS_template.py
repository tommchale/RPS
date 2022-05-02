from pickle import FRAME
import random
import cv2
from keras.models import load_model
import numpy as np
import time
import statistics


class RPS:
    def __init__(self):
        self.rps_options = ['rock', 'paper', 'scissors']
        self.computer_win_count = 0
        self.user_win_count = 0

    def get_computer_choice(self):

        self.computer_choice = random.choice(self.rps_options)

    def get_user_choice(self):

        # loads the tenable model
        model = load_model('keras_model.h5')
        # defines shape of data array
        # an Array of 1 dimensions 224*224*3
        # numpy.ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)[source]
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Captures the video frame by frame
        # opens the camera and defines a capture object
        cap = cv2.VideoCapture(0)

        # Define text details
        text = "Press 'c' to continue..."
        coordinates = (300, 300)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        color = (255, 0, 255)
        thickness = 2

        timeout = time.time() + 5

        predicition_list = []

        while True:

            ret, frame = cap.read()
            frame = cv2.putText(frame, text, coordinates, font,
                                fontScale, color, thickness, cv2.LINE_AA)
            # Resize image
            imS = cv2.resize(frame, (480, 270))
            # shows the image from the frame read by the camera
            cv2.imshow('frame', imS)
            # Press c to continue the window
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break

        timeout = time.time() + 3

        while (timeout - time.time()) > 0:

            ret, frame = cap.read()
            text = (
                f"{timeout - time.time():.0f}")
            frame = cv2.putText(frame, text, coordinates, font,
                                fontScale, color, thickness, cv2.LINE_AA)
            # Resize image
            imS = cv2.resize(frame, (480, 270))
            # shows the image from the frame read by the camera
            cv2.imshow('frame', imS)
            # Press q to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        timeout_2 = time.time() + 1.5

        while time.time() < timeout_2:

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
            # Resize image
            frame = cv2.resize(frame, (480, 270))
            # shows the image from the frame read by the camera
            cv2.imshow('frame', frame)

            prediction = model.predict(data)

            if prediction[0][0] > 0.5:
                predicition_list.append('rock')
                print('Rock')

            elif prediction[0][1] > 0.7:
                predicition_list.append('scissors')
                print('Scissors')

            elif prediction[0][2] > 0.5:
                predicition_list.append('paper')
                print("Paper")

            elif prediction[0][3] > 0.5:
                predicition_list.append('nothing')
                print("Nothing")

        # After the loop release the cap object
        cap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

        self.user_choice = statistics.mode(predicition_list)

    def get_winner(self):

        self.get_computer_choice()
        self.get_user_choice()

        if self.user_choice == 'nothing':
            self.winner = 'nothing'
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')
            print('Please go again and chose Rock, Paper or Scissors')

        if self.computer_choice == self.user_choice:
            self.winner = 'nothing'
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')
            print('draw')

        elif (self.computer_choice == 'paper' and self.user_choice == 'rock') or (self.computer_choice == 'scissors' and self.user_choice == 'paper') or (self.computer_choice == 'rock' and self.user_choice == 'scissors'):
            self.winner = 'computer'
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')
            print('\n The winner is...', self.winner)

        elif self.user_choice != 'nothing':
            self.winner = 'user'
            print(' \n You chose', self.user_choice, ' \n')
            print(' \n The computer chose', self.computer_choice, ' \n')
            print('\n The winner is...', self.winner)

    def play_game(self):
        while True:
            self.get_winner()
            if self.winner == 'computer':
                self.computer_win_count += 1
                print('\n The score is Computer: ',
                      self.computer_win_count, ' - User: ', self.user_win_count)
            elif self.winner == 'user':
                self.user_win_count += 1
                print('\n The score is Computer: ',
                      self.computer_win_count, ' - User: ', self.user_win_count)

            if self.computer_win_count == 3:
                print(' \n The computer has won ! \n')
                break
            elif self.user_win_count == 3:
                print(' \n The user has wone ! \n')
                break


def play():
    game = RPS()
    game.play_game()


play()
