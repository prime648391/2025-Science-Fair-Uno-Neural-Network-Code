import numpy as np
import pickle
import random


# Define the neural network
class SimpleNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.random.randn(hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.random.randn(output_size)

    def printer(self):
        print("bias\n", self.bias1)
        print("weights\n", self.weights1)

    def set_weights_and_bias(self, weights1, bias1, weights2, bias2):
        self.weights1 = weights1
        self.bias1 = bias1
        self.weights2 = weights2
        self.bias2 = bias2

    def weights_and_bias(self):
        return self.weights1, self.bias1, self.weights2, self.bias2

    def forward(self, x):
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        return self.z2

    def backward(self, learning_rate):
        self.weights1 = self.weights1 + np.random.randn(self.input_size, self.hidden_size) * learning_rate
        self.bias1 = self.bias1 + np.random.randn(self.hidden_size) * learning_rate
        self.weights2 = self.weights2 + np.random.randn(self.hidden_size, self.output_size) * learning_rate
        self.bias2 = self.bias2 + np.random.randn(self.output_size) * learning_rate


# Save the network to a file
def save_network(net, filename):
    with open(f"{filename}.pkl", 'wb') as f:
        pickle.dump(net, f)
    # print(f"Network saved to {filename}")


# Load the network from a file
def load_network(filename):
    with open(f"{filename}.pkl", 'rb') as f:
        net = pickle.load(f)
    # print(f"Network loaded from {filename}")
    return net


def train(epochs, batch_size, type, games):
    # Hyperparameters
    input_size = 60
    hidden_size = 20
    output_size = 36
    learning_rate = 5e-3
    num_epochs = epochs

    # Initialize the network
    net = SimpleNet(input_size, hidden_size, output_size)
    net.printer()
    # Training loop
    for epoch in range(num_epochs):
        # print(epoch)
        nets = []
        fitness = [0, 0]
        net_id = 0
        if type == "rand":
            for n in range(0, batch_size):
                nets.append(SimpleNet(input_size, hidden_size, output_size))
                nets[n].set_weights_and_bias(*net.weights_and_bias())
                nets[n].backward(learning_rate)
                save_network(nets[n], "training_rand")
                for g in range(0, games):
                    game = Game(player_1="training_rand", player_2="rand")
                    # Forward pass
                    fitness[n] += game.play()
            net_id = fitness.index(max(*fitness))
        elif type == "self":
            for n in [0, 1]:
                nets.append(SimpleNet(input_size, hidden_size, output_size))
                nets[n].set_weights_and_bias(*net.weights_and_bias())
                nets[n].backward(learning_rate)
                save_network(nets[n], f"training_self_{n}")
            wins = 0
            for n in range(0, games):
                game = Game(player_1="training_self_0", player_2="training_self_1")
                if game.play() > 0:
                    wins += 1
            if wins > games / 2:
                net_id = 0
            else:
                net_id = 1
        net.set_weights_and_bias(*nets[net_id].weights_and_bias())

        # Print progress
        if (epoch + 1) % (epochs / 1000) == 0:
            if type == "rand":
                print(f"Epoch [{epoch + 1}/{num_epochs}], Fitness: {round(100 * fitness[net_id] / games)}")
            else:
                print(f"Epoch [{epoch + 1}/{num_epochs}]")
        if (epoch + 1) % (epochs / 1) == 0:
            net.printer()
    return net


color = ('RED', 'GREEN', 'BLUE', 'YELLOW')
rank = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Skip', 'Reverse', 'Draw2', 'Draw4', 'Wild')
ctype = {'0': 'number', '1': 'number', '2': 'number', '3': 'number', '4': 'number', '5': 'number', '6': 'number',
         '7': 'number', '8': 'number', '9': 'number', 'Skip': 'action', 'Reverse': 'action', 'Draw2': 'action',
         'Draw4': 'action_nocolor', 'Wild': 'action_nocolor'}


class Card:

    def __init__(self, color, rank):
        self.rank = rank
        if ctype[rank] == 'number':
            self.color = color
            self.cardtype = 'number'
        elif ctype[rank] == 'action':
            self.color = color
            self.cardtype = 'action'
        else:
            self.color = None
            self.cardtype = 'action_nocolor'

    def __str__(self):
        if self.color == None:
            return self.rank
        else:
            return self.color + " " + self.rank


class Deck:

    def __init__(self):
        self.deck = []
        for clr in color:
            for ran in rank:
                if ctype[ran] != 'action_nocolor':
                    self.deck.append(Card(clr, ran))
                    self.deck.append(Card(clr, ran))
                else:
                    self.deck.append(Card(clr, ran))

    def __str__(self):
        deck_comp = ''
        for card in self.deck:
            deck_comp += '\n' + card.__str__()
        return 'The deck has ' + deck_comp

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self):
        if len(self.deck) > 0:
            return self.deck.pop()
        else:
            self.__init__()
            return self.deal()


class Hand:

    def __init__(self):
        self.cards = []
        self.cardsstr = []
        self.number_cards = 0
        self.action_cards = 0

    def add_card(self, card):
        self.cards.append(card)
        self.cardsstr.append(str(card))
        if card.cardtype == 'number':
            self.number_cards += 1
        else:
            self.action_cards += 1

    def remove_card(self, place):
        self.cardsstr.pop(int(place) - 1)
        return self.cards.pop(int(place) - 1)

    def cards_in_hand(self):
        for i in range(len(self.cardsstr)):
            print(f' {i + 1}.{self.cardsstr[i]}')
            pass

    def single_card(self, place):
        return self.cards[int(place) - 1]

    def no_of_cards(self):
        return len(self.cards)


# Function to randomly select who starts first
def choose_first():
    if random.randint(0, 1) == 0:
        return 'Player'
    else:
        return 'Pc'


# Function to check if the card thrown by Player/PC is a valid card by comparing it with the top card
def single_card_check(top_card, card):
    if card.color == top_card.color or top_card.rank == card.rank or card.cardtype == 'action_nocolor':
        return True
    else:
        return False


# FOR PC ONLY
# To check if PC has any valid card to throw
def full_hand_check(hand, top_card):
    for c in hand.cards:
        if c.color == top_card.color or c.rank == top_card.rank or c.cardtype == 'action_nocolor':
            return hand.remove_card(hand.cardsstr.index(str(c)) + 1)
    else:
        return 'no card'


# Function to check if either wins
def win_check(hand):
    if len(hand.cards) == 0 or len(hand.cards) >= 30:
        return True
    else:
        return False


# Function to check if last card is an action card (GAME MUST END WITH A NUMBER CARD)
def last_card_check(hand):
    for c in hand.cards:
        if c.cardtype != 'number':
            return True
        else:
            return False


def determine_output(np_outputs, goal):
    outputs = list(np_outputs)
    if goal == "hit":
        if outputs[34] > outputs[35]:
            return "h"
        else:
            return "p"
    elif goal == "color":
        colors = [outputs[i + 30] for i in range(0, 3)]
        return ["RED", "YELLOW", "GREEN", "BLUE"][colors.index(max(*colors))]
    else:
        if outputs.index(max(*outputs)) > goal:
            return "1"
        else:
            return str(outputs.index(max(*outputs)))


def reshape_hand(hand):
    results = []
    for i in range(0, min(len(hand.cards), 30)):
        results.append(hand.cards[i].color)
        results.append(hand.cards[i].rank)
    for n in range(0, 30 - len(hand.cards)):
        results.append("None")
        results.append("None")
    return str_to_num(results)


def str_to_num(list):
    terms = {"None": 0, "RED": 0.25, "YELLOW": 0.5, "GREEN": 0.75, "BLUE": 1, "0": 1 / 15, "1": 2 / 15, "2": 3 / 15,
             "3": 4 / 15, "4": 5 / 15, "5": 6 / 15, "6": 7 / 15, "7": 8 / 15, "8": 9 / 15, "9": 10 / 15,
             "Skip": 11 / 15, "Reverse": 12 / 15, "Draw2": 13 / 15, "Draw4": 14 / 15, "Wild": 1, None: 0}
    return [terms[N] for N in list]


def calculate_hand(hand):
    hand_value = 0
    for card in hand.cards:
        hand_value += int({"number": card.rank, "action": "20", "action_nocolor": "50"}[card.cardtype])
    return hand_value


# The gaming loop
class Game:
    def __init__(self, player_1, player_2):
        self.winner = 0
        self.player_1 = player_1
        self.player_2 = player_2
        if self.player_1 == "human":
            print('Welcome to UNO! Finish your cards first to win')
        elif not type(player_1) == str:
            self.main_net_input_data = player_1
        else:
            self.main_net = load_network(f"{self.player_1}")
        if self.player_2 != "rand":
            if type(player_1) == str:
                self.net = load_network(f"{self.player_2}")
            else:
                self.net = player_2
        self.deck = Deck()
        self.deck.shuffle()

        self.player_hand = Hand()
        for i in range(7):
            self.player_hand.add_card(self.deck.deal())

        self.pc_hand = Hand()
        for i in range(7):
            self.pc_hand.add_card(self.deck.deal())

        self.top_card = self.deck.deal()
        if self.top_card.cardtype != 'number':
            while self.top_card.cardtype != 'number':
                self.top_card = self.deck.deal()
        if self.player_1 == "human":
            print('\nStarting Card is: {}'.format(self.top_card))

        self.turn = choose_first()
        if player_1 == "human":
            print(self.turn + ' will go first')
        self.playing = False
        self.main_net_input_data = reshape_hand(self.player_hand)
        self.net_input_data = reshape_hand(self.pc_hand)

    def play(self):
        self.playing = True
        while self.playing:
            if self.player_1 != "human":
                self.p1_results = self.main_net.forward(reshape_hand(self.player_hand))
            if self.player_2 != "rand":
                self.p2_results = self.net.forward(reshape_hand(self.pc_hand))
            if self.turn == 'Player':
                if self.player_1 == "human":
                    print('\nTop card is: ' + str(self.top_card))
                    print('Your cards: ')
                    self.player_hand.cards_in_hand()
                if self.player_1 == "human":
                    self.choice = input("\nHit or Pull? (h/p): ")
                else:
                    self.choice = determine_output(self.p1_results, "hit")
                if self.choice == 'h':
                    if self.player_1 == "human":
                        self.pos = int(input('Enter index of card: '))
                    else:
                        self.pos = determine_output(self.p1_results, len(self.player_hand.cards))
                    self.temp_card = self.player_hand.single_card(self.pos)
                    if single_card_check(self.top_card, self.temp_card):
                        if self.temp_card.cardtype == 'number':
                            self.top_card = self.player_hand.remove_card(self.pos)
                            self.turn = 'Pc'
                        else:
                            if self.temp_card.rank == 'Skip':
                                self.turn = 'Player'
                                self.top_card = self.player_hand.remove_card(self.pos)
                            elif self.temp_card.rank == 'Reverse':
                                self.turn = 'Player'
                                self.top_card = self.player_hand.remove_card(self.pos)
                            elif self.temp_card.rank == 'Draw2':
                                self.pc_hand.add_card(self.deck.deal())
                                self.pc_hand.add_card(self.deck.deal())
                                self.top_card = self.player_hand.remove_card(self.pos)
                                self.turn = 'Player'
                            elif self.temp_card.rank == 'Draw4':
                                for i in range(4):
                                    self.pc_hand.add_card(self.deck.deal())
                                self.top_card = self.player_hand.remove_card(self.pos)
                                if self.player_1 == "human":
                                    self.draw4color = input('Change color to (enter in caps): ')
                                else:
                                    self.draw4color = determine_output(self.p1_results, "color")
                                if self.draw4color != self.draw4color.upper():
                                    self.draw4color = self.draw4color.upper()
                                self.top_card.color = self.draw4color
                                self.turn = 'Player'
                            elif self.temp_card.rank == 'Wild':
                                self.top_card = self.player_hand.remove_card(self.pos)
                                if self.player_1 == "human":
                                    self.wildcolor = input('Change color to (enter in caps): ')
                                else:
                                    self.wildcolor = determine_output(self.p1_results, "color")
                                if self.wildcolor != self.wildcolor.upper():
                                    self.wildcolor = self.wildcolor.upper()
                                self.top_card.color = self.wildcolor

                                self.turn = 'Pc'
                    else:
                        if self.player_1 == "human":
                            print('This card cannot be used')
                        self.choice = "p"
                if self.choice == 'p':
                    self.temp_card = self.deck.deal()
                    if self.player_1 == "human":
                        print('You got: ' + str(self.temp_card))
                    if single_card_check(self.top_card, self.temp_card):
                        self.player_hand.add_card(self.temp_card)
                    else:
                        if self.player_1 == "human":
                            print('Cannot use this card')
                        self.player_hand.add_card(self.temp_card)
                    self.turn = 'Pc'
                if win_check(self.player_hand):
                    if len(self.player_hand.cards) >= 30:
                        # print("Player has to many cards, loser")
                        self.winner = -1 * calculate_hand(self.player_hand)
                    else:
                        # print('\nPLAYER WON!!')
                        self.winner = calculate_hand(self.pc_hand)
                    self.playing = False
                    break

            if self.turn == 'Pc':
                if self.player_2 == "rand":
                    if self.pc_hand.no_of_cards() == 1:
                        if last_card_check(self.pc_hand):
                            if self.player_1 == "human":
                                print('Adding a card to PC hand')
                            self.pc_hand.add_card(self.deck.deal())
                    self.temp_card = full_hand_check(self.pc_hand, self.top_card)
                    if self.temp_card != 'no card':
                        if self.player_1 == "human":
                            print(f'\nPC throws: {self.temp_card}')
                        if self.temp_card.cardtype == 'number':
                            self.top_card = self.temp_card
                            self.turn = 'Player'
                        else:
                            if self.temp_card.rank == 'Skip':
                                self.turn = 'Pc'
                                self.top_card = self.temp_card
                            elif self.temp_card.rank == 'Reverse':
                                self.turn = 'Pc'
                                self.top_card = self.temp_card
                            elif self.temp_card.rank == 'Draw2':
                                self.player_hand.add_card(self.deck.deal())
                                self.player_hand.add_card(self.deck.deal())
                                self.top_card = self.temp_card
                                self.turn = 'Pc'
                            elif self.temp_card.rank == 'Draw4':
                                for i in range(4):
                                    self.player_hand.add_card(self.deck.deal())
                                self.top_card = self.temp_card
                                self.draw4color = self.pc_hand.cards[0].color
                                if self.player_1 == "human":
                                    print('Color changes to', self.draw4color)
                                self.top_card.color = self.draw4color
                                self.turn = 'Pc'
                            elif self.temp_card.rank == 'Wild':
                                self.top_card = self.temp_card
                                self.wildcolor = self.pc_hand.cards[0].color
                                if self.player_1 == "human":
                                    print("Color changes to", self.wildcolor)
                                self.top_card.color = self.wildcolor
                                self.turn = 'Player'
                    else:
                        if self.player_1 == "human":
                            print('\nPC pulls a card from deck')
                        self.temp_card = self.deck.deal()
                        if single_card_check(self.top_card, self.temp_card):
                            if self.player_1 == "human":
                                print(f'PC throws: {self.temp_card}')
                            if self.temp_card.cardtype == 'number':
                                self.top_card = self.temp_card
                                self.turn = 'Player'
                            else:
                                if self.temp_card.rank == 'Skip':
                                    self.turn = 'Pc'
                                    self.top_card = self.temp_card
                                elif self.temp_card.rank == 'Reverse':
                                    self.turn = 'Pc'
                                    self.top_card = self.temp_card
                                elif self.temp_card.rank == 'Draw2':
                                    self.player_hand.add_card(self.deck.deal())
                                    self.player_hand.add_card(self.deck.deal())
                                    self.top_card = self.temp_card
                                    self.turn = 'Pc'
                                elif self.temp_card.rank == 'Draw4':
                                    for i in range(4):
                                        self.player_hand.add_card(self.deck.deal())
                                    self.top_card = self.temp_card
                                    self.draw4color = self.pc_hand.cards[0].color
                                    if self.player_1 == "human":
                                        print('Color changes to', self.draw4color)
                                    self.top_card.color = self.draw4color
                                    self.turn = 'Pc'
                                elif self.temp_card.rank == 'Wild':
                                    self.top_card = self.temp_card
                                    self.wildcolor = self.pc_hand.cards[0].color
                                    if self.player_1 == "human":
                                        print('Color changes to', self.wildcolor)
                                    self.top_card.color = self.wildcolor
                                    self.turn = 'Player'
                        else:
                            if self.player_1 == "human":
                                print('PC doesnt have a card')
                            self.pc_hand.add_card(self.temp_card)
                            self.turn = 'Player'
                else:
                    if self.player_1 == "human":
                        pass
                        # self.pc_hand.cards_in_hand()
                    self.choice = determine_output(self.p2_results, "hit")
                    if self.choice == 'h':
                        self.pos = determine_output(self.p2_results, len(self.pc_hand.cards))
                        self.temp_card = self.pc_hand.single_card(self.pos)
                        if single_card_check(self.top_card, self.temp_card):
                            if self.player_1 == "human":
                                print(f"PC plays {self.temp_card.__str__()}")
                            if self.temp_card.cardtype == 'number':
                                self.top_card = self.pc_hand.remove_card(self.pos)
                                self.turn = 'Player'
                            else:
                                if self.temp_card.rank == 'Skip':
                                    self.turn = 'Pc'
                                    self.top_card = self.pc_hand.remove_card(self.pos)
                                elif self.temp_card.rank == 'Reverse':
                                    self.turn = 'Pc'
                                    self.top_card = self.pc_hand.remove_card(self.pos)
                                elif self.temp_card.rank == 'Draw2':
                                    self.pc_hand.add_card(self.deck.deal())
                                    self.pc_hand.add_card(self.deck.deal())
                                    self.top_card = self.pc_hand.remove_card(self.pos)
                                    self.turn = 'Pc'
                                elif self.temp_card.rank == 'Draw4':
                                    for i in range(4):
                                        self.player_hand.add_card(self.deck.deal())
                                    self.top_card = self.pc_hand.remove_card(self.pos)
                                    self.draw4color = determine_output(self.p2_results, "color")
                                    if self.draw4color != self.draw4color.upper():
                                        self.draw4color = self.draw4color.upper()
                                    self.top_card.color = self.draw4color
                                    self.turn = 'Pc'
                                elif self.temp_card.rank == 'Wild':
                                    self.top_card = self.pc_hand.remove_card(self.pos)
                                    self.wildcolor = determine_output(self.p2_results, "color")
                                    if self.wildcolor != self.wildcolor.upper():
                                        self.wildcolor = self.wildcolor.upper()
                                    self.top_card.color = self.wildcolor

                                    self.turn = 'Player'
                        else:
                            self.choice = "p"
                    if self.choice == 'p':
                        if self.player_1 == "human":
                            print("PC pulls card from deck")
                        self.temp_card = self.deck.deal()
                        if single_card_check(self.top_card, self.temp_card):
                            self.pc_hand.add_card(self.temp_card)
                        else:
                            self.pc_hand.add_card(self.temp_card)
                        self.turn = 'Player'
                if self.player_1 == "human":
                    print('\nPC has {} cards remaining'.format(self.pc_hand.no_of_cards()))
                if win_check(self.pc_hand):
                    if len(self.pc_hand.cards) >= 30:
                        # print("Pc has too many cards, loser")
                        self.winner = calculate_hand(self.pc_hand)
                    else:
                        # print('\nPC WON!!')
                        self.winner = -1 * calculate_hand(self.player_hand)
                    self.playing = False
                    break
        # print("score\n", winner)
        return self.winner / 100


# save_network(train(1000, 2, "rand", games=127), "rand")
game = Game("human", "rand")
game.play()
