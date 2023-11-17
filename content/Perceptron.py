import ipywidgets as widgets
import pandas as pd
from ipywidgets import Layout
from IPython.display import display
from random import random, shuffle
from enum import Enum, auto
import gettext

gettext.bindtextdomain('perceptron', 'translations')
gettext.textdomain('perceptron')
_ = gettext.gettext

# column names in training protocol
INPUT_1_COLUMN = "$i_1$"
INPUT_2_COLUMN = "$i_2$"
WEIGHT_0_COLUMN = "$w_0$"
WEIGHT_1_COLUMN = "$w_1$"
WEIGHT_2_COLUMN = "$w_2$"
TARGET_COLUMN = "$t$"
OUTPUT_COLUMN = "$o$"
DELTA_COLUMN = "$\delta$"


class Perceptron:

    training = [
        [1, 0, 0],  # no weekend, no parents
        [1, 0, 1],  # no weekend, parents
        [1, 1, 0],  # weekend, no parents
        [1, 1, 1]]  # weekend, parents
    weights = [0, 0, 0]
    targets = None
    trainingProtocol = None
    currentIndex = 0
    delta = None
    error = None
    state = None
    epoch = None
    indexes = [i for i in range(0, len(training))]
    alpha = 0.1

    class State(Enum):
        START = auto()
        GET_INPUT = auto()
        CALCULATE_OUTPUT = auto()
        RECALIBRATE = auto()

    def __init__(self):

        # create GUI elements for theoretical input
        theory1 = widgets.HTML(
            _("<h1>Theory</h1>") +
            _("<h2>Why machine learning?</h2>") +
            _("""
            <p>In the past, we typically used a programming language such as C,
            Java or Python to solve problems with computers, using explicit
            commands to create the data structures, algorithms, functions and
            ultimately executable programs required for the solution. This
            approach can be used to create excellent solutions in many areas.
            However, in complex situations, such as large amounts of data or
            dynamically changing environments, we reach the limits of
            conventional programming, e.g. in speech recognition, autonomous
            driving or medical diagnostics.</p>
            """) +
            _("""
            <p>In order to efficiently create solutions for these complex
            applications, the field of machine learning has established itself
            as a supplement to traditional programming. As is so often the
            case, the model for machine learning is taken from nature: the
            brain with its network of neurons. In order to understand the
            mathematical model based on this, let's first take a look at how a
            biological neuron works:</p>
            """) +
            _("<h2>Biological neuron</h2>")
        )

        # The HTML widget from ipywidgets can't resolve relative file names.
        # Therefore we use the Image widget here in between.
        imagePath = "pictures/" + _("neuron_en.png")
        neuronImage = widgets.Image(
            value=open(imagePath, "rb").read(), width=400)

        theory2 = widgets.HTML(
            _("""
            <p>In simple terms, biological neurons function by receiving and
            transmitting electrical impulses. Reception takes place via the
            dentrites. The potentials of the electrical impulses received are
            summed up in the cell body (soma). As soon as this sum exceeds a
            certain threshold value, the action potential, the electrical
            impulse is transmitted on the axon. This works according to the
            all-or-nothing principle: as long as the action potential is not
            reached, no impulse is transmitted at all; as soon as it is
            reached, the impulse is transmitted. The closer a dendrite is to
            the axon, the stronger the influence of its incoming impulse on the
            action potential in the cell body. Its proximity to the axon
            therefore influences the "weighting" of a dendrite. Transmitted
            impulses are transferred to other cells (e.g. other neurons, gland
            or muscle cells) at the axon terminals.</p>
            """) +
            _("""
            <p>The almost 100 billion neurons in our human brain each have
            around 10,000 dendrites, enabling us to perform our outstanding
            learning and thinking abilities.</p>
            """) +
            _("""
            <p>The simplified model of a biological neuron described above can
            now be converted into a mathematical model:</p>
            """) +
            _("<h2>Artificial neuron</h2>")
        )

        imagePath = "pictures/" + _("artificial_neuron_en.png")
        artificialNeuronImage = widgets.Image(
            value=open(imagePath, "rb").read(), width=600)

        theory3 = widgets.HTML(
            _("""
            <p>
            <ul>
                <li>
                    the dentrites become the inputs x<sub>1</sub> to
                    x<sub>n</sub>
                </li>
                <li>
                    the proximity of the dentrites to the axon is indicated by
                    the respective weights w<sub>1</sub> to w<sub>n</sub>
                </li>
                <li>
                    the summation of the received electrical impulses in the
                    cell body is mapped in the transfer function
                </li>
                <li>
                    the action potential is stored in the threshold value
                    &theta;
                </li>
                <li>
                    the all-or-nothing principle is represented by the
                    activation function
                </li>
            </ul>
            </p>
            """) +
            _("""
            <p>But how does an artificial neuron "learn"? Let's try it with a
            very simple example...</p>
            """) +
            _("<h2>Example</h2>") +
            _("""
            <p>In our example, there should be exactly two inputs, each of which
            can only accept truth values (i.e. "true" or "false"). The two
            inputs are:
            <ul>
                <li>It is weekend.</li>
                <li>Parents come to visit.</li>
            </ul>
            We want to use these two inputs to determine whether it is a good
            day:</p>
            """)
        )

        imagePath = "pictures/" + _("good_day_empty_en.png")
        goodDayEmptyImage = widgets.Image(
            value=open(imagePath, "rb").read(), width=550)

        theory4 = widgets.HTML(
            _("""
            <p>As you can easily imagine, the answer to this question is very
            individual. Let's assume we are looking at the situation for a
            person for whom it is always a good day when it is weekend or
            their parents are visiting, or both:</p>
            """)
        )

        imagePath = "pictures/" + _("good_day_example_en.png")
        goodDayExampleImage = widgets.Image(
            value=open(imagePath, "rb").read(), width=550)

        theory5 = widgets.HTML(
            _("""
            <p>If you have ever programmed before, you will most likely
            immediately see that we are dealing with an OR operation of the two
            inputs. But for once, we do <em>NOT</em> want to solve this task
            with explicit commands, but rather teach an artificial neuron to
            always generate the desired output for all possible variations of
            the inputs.</p>
            """) +
            _("""
            <p>To do this, we actually just have to set all the
            <span style="color:red">weights</span> and the
            <span style="color:red">threshold value</span> in our artificial
            neuron appropriately:</p>
            """)
        )

        imagePath = "pictures/" + _("simple_neuron_en.png")
        simpleNeuronImage = widgets.Image(
            value=open(imagePath, "rb").read(), width=600)

        theoryBox = widgets.VBox([
            theory1, neuronImage, theory2, artificialNeuronImage, theory3,
            goodDayEmptyImage, theory4, goodDayExampleImage, theory5,
            simpleNeuronImage
        ])

        # create GUI elements for training
        trainingHeader = widgets.HTML(_(
            """
            <h1>Training</h1>
            In this area the perceptron can be trained:
            <ol>
                <li>
                    Select the checkboxes for the combinations you consider a
                    nice day in the rightmost column.
                </li>
                <li>Press the start button to start the learning process.</li>
                <li>
                    Press the "Finalize Epoch" button to continue learning
                    until the current epoch is finished.
                </li>
                <li>
                    Press the "Reset" button to reset the learning process.
                </li>
            </ol>
            """
        ))

        # we don't need the empty description label in front of the checkboxes
        checkBoxStyle = {'description_width': '0px'}
        checkBoxLayout = Layout(width="fit-content")
        vBoxLayout = Layout(
            flex='0 0 auto', width="fit-content",
            margin="0px 30px 0px 0px", align_items="center")
        targetLayout = Layout(
            flex='0 0 auto', width="fit-content",
            margin="0px 30px 0px 0px", align_items="center",
            border_left="1px solid", padding="0px 0px 0px 30px")

        inputImage1 = widgets.Image(
            value=open("pictures/weekend.png", "rb").read(), width=100)
        inputLabel1 = widgets.Label(value=_("Weekend"))
        self.inputCheckBox1A = widgets.Checkbox(
            disabled=True, style=checkBoxStyle,
            layout=checkBoxLayout, value=False)
        self.inputCheckBox1B = widgets.Checkbox(
            disabled=True, style=checkBoxStyle,
            layout=checkBoxLayout, value=False)
        self.inputCheckBox1C = widgets.Checkbox(
            disabled=True, style=checkBoxStyle,
            layout=checkBoxLayout, value=True)
        self.inputCheckBox1D = widgets.Checkbox(
            disabled=True, style=checkBoxStyle,
            layout=checkBoxLayout, value=True)
        input1VBox = widgets.VBox([inputImage1, inputLabel1,
                                   self.inputCheckBox1A, self.inputCheckBox1B,
                                   self.inputCheckBox1C, self.inputCheckBox1D])
        input1VBox.layout = vBoxLayout

        inputImage2 = widgets.Image(
            value=open("pictures/parents.png", "rb").read(), width=100)
        inputLabel2 = widgets.Label(value=_("Visiting Parents"))
        self.inputCheckBox2A = widgets.Checkbox(
            disabled=True, style=checkBoxStyle,
            layout=checkBoxLayout, value=False)
        self.inputCheckBox2B = widgets.Checkbox(
            disabled=True, style=checkBoxStyle,
            layout=checkBoxLayout, value=True)
        self.inputCheckBox2C = widgets.Checkbox(
            disabled=True, style=checkBoxStyle,
            layout=checkBoxLayout, value=False)
        self.inputCheckBox2D = widgets.Checkbox(
            disabled=True, style=checkBoxStyle,
            layout=checkBoxLayout, value=True)
        input2VBox = widgets.VBox([inputImage2, inputLabel2,
                                   self.inputCheckBox2A, self.inputCheckBox2B,
                                   self.inputCheckBox2C, self.inputCheckBox2D])
        input2VBox.layout = vBoxLayout

        targetImage = widgets.Image(
            value=open("pictures/happy.png", "rb").read(), width=100)
        targetLabel = widgets.Label(value=_("Nice Day?"))
        self.targetCheckBoxA = widgets.Checkbox(
            style=checkBoxStyle, layout=checkBoxLayout, value=False)
        self.targetCheckBoxB = widgets.Checkbox(
            style=checkBoxStyle, layout=checkBoxLayout, value=True)
        self.targetCheckBoxC = widgets.Checkbox(
            style=checkBoxStyle, layout=checkBoxLayout, value=True)
        self.targetCheckBoxD = widgets.Checkbox(
            style=checkBoxStyle, layout=checkBoxLayout, value=True)
        targetVBox = widgets.VBox([targetImage, targetLabel,
                                   self.targetCheckBoxA, self.targetCheckBoxB,
                                   self.targetCheckBoxC, self.targetCheckBoxD])
        targetVBox.layout = targetLayout

        targetsBox = widgets.HBox([input1VBox, input2VBox, targetVBox])

        self.alphaText = widgets.FloatText(
            description=_("Learning Rate α:"), value=self.alpha,
            layout=Layout(margin="30px 0px 0px 10px", width="fit-content"),
            style=dict(description_width='initial'))

        self.stepButton = widgets.Button(
            layout=Layout(margin="0px 0px 0px 0px", width="fit-content"))

        self.finishEpochButton = widgets.Button(
            description=_("Finalize Epoch"),
            layout=Layout(margin="0px 0px 0px 10px", width="fit-content"))

        self.resetButton = widgets.Button(
            description=_("Reset"),
            layout=Layout(margin="0px 0px 0px 10px", width="fit-content"))

        self.epochLabel = widgets.Label(
            layout=Layout(margin="0px 0px 0px 10px"))

        buttonsBox = widgets.HBox(
            [self.stepButton, self.finishEpochButton, self.resetButton],
            layout=Layout(margin="10px 0px 0px 10px"))

        self.protocolOutput = widgets.Output()

        # create GUI elements for inference
        inferenceHeader = widgets.HTML(_(
            """
            <h1>Inference</h1>
            In this area the perceptron can be used:
            <ol>
                <li>
                    Change the selection of the two input checkboxes on the
                    left hand side.
                </li>
                <li>Watch the result on the righthand side.</li>
            </ol>
            """
        ))

        self.inputCheckBox1 = widgets.Checkbox(
            style=checkBoxStyle, layout=checkBoxLayout)
        input1VBox = widgets.VBox(
            [inputImage1, inputLabel1, self.inputCheckBox1])
        input1VBox.layout = vBoxLayout

        self.inputCheckBox2 = widgets.Checkbox(
            style=checkBoxStyle, layout=checkBoxLayout)
        input2VBox = widgets.VBox(
            [inputImage2, inputLabel2, self.inputCheckBox2])
        input2VBox.layout = vBoxLayout

        self.targetCheckBox = widgets.Checkbox(
            disabled=True, style=checkBoxStyle, layout=checkBoxLayout)
        targetVBox = widgets.VBox(
            [targetImage, targetLabel, self.targetCheckBox])
        targetVBox.layout = targetLayout

        trainingBox = widgets.VBox(
            [trainingHeader, targetsBox, self.alphaText,
             buttonsBox, self.epochLabel, self.protocolOutput])

        inferenceBox = widgets.VBox(
            [inferenceHeader,
             widgets.HBox([input1VBox, input2VBox, targetVBox])])

        self.GUI = widgets.VBox([theoryBox, trainingBox, inferenceBox])

        self.switchTo(Perceptron.State.START)

    def scalarProduct(self, list1, list2):
        result = 0
        for i in range(len(list1)):
            result = result + (list1[i] * list2[i])
        return result

    def heaviside(self, x):
        if x < 0:
            return 0
        else:
            return 1

    def switchTo(self, newState):
        if newState == Perceptron.State.START:
            self.stepButton.description = _("Start (set random weights))")
            self.clearBackground()
            self.setTargetEnabled(True)
            self.updateProtocol()
        elif newState == Perceptron.State.GET_INPUT:
            self.stepButton.description = _("Continue (get random input)")
        elif newState == Perceptron.State.CALCULATE_OUTPUT:
            self.stepButton.description = _("Continue (calculate result)")
        elif newState == Perceptron.State.RECALIBRATE:
            self.stepButton.description = _("Continue (adopt weights)")
        self.state = newState

    def setBackground(self, color, index):
        if index == 0:
            self.inputCheckBox1A.style.background = color
            self.inputCheckBox2A.style.background = color
            self.targetCheckBoxA.style.background = color
        elif index == 1:
            self.inputCheckBox1B.style.background = color
            self.inputCheckBox2B.style.background = color
            self.targetCheckBoxB.style.background = color
        elif index == 2:
            self.inputCheckBox1C.style.background = color
            self.inputCheckBox2C.style.background = color
            self.targetCheckBoxC.style.background = color
        elif index == 3:
            self.inputCheckBox1D.style.background = color
            self.inputCheckBox2D.style.background = color
            self.targetCheckBoxD.style.background = color

    def clearBackground(self):
        for i in range(0, 4):
            self.setBackground('white', i)

    def addRow(self):
        new_row = pd.DataFrame({
            INPUT_1_COLUMN: [''],
            INPUT_2_COLUMN: [''],
            WEIGHT_0_COLUMN: [self.weights[0]],
            WEIGHT_1_COLUMN: [self.weights[1]],
            WEIGHT_2_COLUMN: [self.weights[2]],
            TARGET_COLUMN: [''],
            OUTPUT_COLUMN: [''],
            DELTA_COLUMN: ['']
        })

        self.trainingProtocol = pd.concat(
            [self.trainingProtocol, new_row], ignore_index=True)

    def startEpoch(self):
        self.clearBackground()
        shuffle(self.indexes)  # prevent oszillation
        self.currentIndex = 0
        self.epoch += 1
        self.error = False
        self.addRow()
        self.epochLabel.value = _("Epoch: ") + str(self.epoch)
        self.switchTo(Perceptron.State.GET_INPUT)

    def setTargetEnabled(self, enabled):
        self.targetCheckBoxA.disabled = not enabled
        self.targetCheckBoxB.disabled = not enabled
        self.targetCheckBoxC.disabled = not enabled
        self.targetCheckBoxD.disabled = not enabled

    def getInput(self):
        tmpIndex = self.indexes[self.currentIndex]
        protocolIndex = self.trainingProtocol.index[-1]
        self.trainingProtocol.loc[protocolIndex, INPUT_1_COLUMN] = (
            self.training[tmpIndex][1])
        self.trainingProtocol.loc[protocolIndex, INPUT_2_COLUMN] = (
            self.training[tmpIndex][2])
        self.trainingProtocol.loc[protocolIndex, TARGET_COLUMN] = (
            self.targets[tmpIndex])
        self.setBackground('yellow', tmpIndex)
        self.switchTo(Perceptron.State.CALCULATE_OUTPUT)

    def calculateOutput(self):
        tmpIndex = self.indexes[self.currentIndex]
        output = self.heaviside(self.scalarProduct(
            self.training[tmpIndex], self.weights))
        self.delta = self.targets[self.indexes[self.currentIndex]] - output
        protocolIndex = self.trainingProtocol.index[-1]
        self.trainingProtocol.loc[protocolIndex, OUTPUT_COLUMN] = output
        self.trainingProtocol.loc[protocolIndex, DELTA_COLUMN] = self.delta
        if self.delta == 0:
            self.setBackground('green', tmpIndex)
        else:
            self.setBackground('red', tmpIndex)
            self.error = True
        self.switchTo(Perceptron.State.RECALIBRATE)

    def recalibrate(self):
        tmpIndex = self.indexes[self.currentIndex]

        for n in range(0, len(self.weights)):
            self.weights[n] = (
                self.weights[n] +
                self.delta * self.alpha * self.training[tmpIndex][n])
            self.updateTargetCheckbox()

        if self.currentIndex == len(self.indexes) - 1:
            # end of epoch reached
            if self.error:
                self.startEpoch()
            else:
                self.switchTo(Perceptron.State.START)
                self.stepButton.disabled = True
                self.finishEpochButton.disabled = True
        else:
            self.addRow()
            self.currentIndex += 1
            self.switchTo(Perceptron.State.GET_INPUT)

    def runStep(self):
        if self.state == Perceptron.State.START:
            # get target values from user defined checkboxes
            self.setTargetEnabled(False)
            self.targets = [1 if self.targetCheckBoxA.value else 0,
                            1 if self.targetCheckBoxB.value else 0,
                            1 if self.targetCheckBoxC.value else 0,
                            1 if self.targetCheckBoxD.value else 0]

            self.trainingProtocol = pd.DataFrame({
                INPUT_1_COLUMN: [],
                INPUT_2_COLUMN: [],
                WEIGHT_0_COLUMN: [],
                WEIGHT_1_COLUMN: [],
                WEIGHT_2_COLUMN: [],
                TARGET_COLUMN: [],
                OUTPUT_COLUMN: [],
                DELTA_COLUMN: []
            })

            # start with random weights
            self.weights = [random() for i in range(0, 3)]
            self.epoch = 0
            self.startEpoch()

        elif self.state == Perceptron.State.GET_INPUT:
            self.getInput()

        elif self.state == Perceptron.State.CALCULATE_OUTPUT:
            self.calculateOutput()

        elif self.state == Perceptron.State.RECALIBRATE:
            self.recalibrate()

        self.updateProtocol()

    def updateProtocol(self):
        if self.trainingProtocol is None:
            self.protocolOutput.clear_output()
        else:
            # start index with 1 instead of 0 for better readability
            self.trainingProtocol.index = (
                range(1, len(self.trainingProtocol) + 1))
            coloredProtocol = self.trainingProtocol.tail(8).style.apply(
                highlightProtocol, axis='columns')
            # Rendering in JupyterLab is very smooth. Unfortunately, in
            # JupyterLite the screen flickers and the data frame fills the
            # whole screen width. Therefore we manually set the column widths
            # here.
            # startof JupyterLite workaround
            coloredProtocol.set_properties(
                subset=[INPUT_1_COLUMN, INPUT_2_COLUMN, TARGET_COLUMN,
                        OUTPUT_COLUMN, DELTA_COLUMN],
                **{'width': '20px'})
            coloredProtocol.set_properties(
                subset=[WEIGHT_0_COLUMN, WEIGHT_1_COLUMN, WEIGHT_2_COLUMN],
                **{'width': '60px'})
            # endof JupyterLite workaround

            self.protocolOutput.clear_output(wait=True)
            with self.protocolOutput:
                display(coloredProtocol)

    def updateTargetCheckbox(self):
        inputs = [1,
                  1 if self.inputCheckBox1.value else 0,
                  1 if self.inputCheckBox2.value else 0]
        output = self.heaviside(self.scalarProduct(inputs, self.weights))
        self.targetCheckBox.value = 0 if output == 0 else 1

    def endOfEpochReached(self):
        return (self.currentIndex == len(self.indexes) - 1
                and self.state == Perceptron.State.RECALIBRATE)

    def finishEpoch(self):
        if self.endOfEpochReached():
            self.runStep()
            if self.state == Perceptron.State.START:
                return  # learning finished
        while not self.endOfEpochReached():
            self.runStep()

    def updateAlpha(self):
        self.alpha = self.alphaText.value

    def reset(self):
        self.trainingProtocol = None
        self.switchTo(Perceptron.State.START)
        self.stepButton.disabled = False
        self.finishEpochButton.disabled = False
        self.epochLabel.value = _("Epoch: ")


def highlightProtocol(series):
    input1 = series.loc[INPUT_1_COLUMN]
    if input1 == '':
        # fresh line (only weights)
        return [''] * series.size

    delta = series.loc[DELTA_COLUMN]
    if delta == '':
        # added inputs and targets
        color = 'background-color: yellow'
        return [color] * 2 + [''] * 3 + [color] + [''] * 2
    elif delta == 0:
        color = 'background-color: green'
    elif delta == 1 or delta == -1:
        color = 'background-color: red'
    else:
        color = ''
    return [color] * series.size