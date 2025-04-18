��    7      �  I   �      �  w  �  q  )  �  �  �    �  �  �  �  J    u   c  �  �  %  �  �   �  �   �  �   
  s   �  �  T  �  �  �  �  %  �   �  �$  �  d'  �   B)  �  �)  �   K+     !,  �  B-  Y   $0  �   ~0  �   1  �   �1  [   �2  ~  *3     �4     �4     �4     �4      5     5     95     R5     n5     �5     �5     �5  	   �5     �5     �5     �5     �5     �5     6     '6  '   ?6     g6     u6    �6  �  �7  �  o:  a  �;  �  R?  �  ,B  �  �C    kG  u   �H  �  �H  '  �J  �   �K  �   �L  �   M  n   N  �  wN    .P  �  1R  �  �S  �  �W  �  8Z  �   �[  S  d\  �   �]    �^  �  �_  W   �b  �   c  �   �c  �   �d  k   =e  c  �e     g      g      ?g     `g     rg  #   �g     �g     �g      �g     h     h     +h     7h     Eh  &   Sh     zh  
   �h     �h     �h     �h  '   �h     i     i                7          3      5         2      6      	   ,   #                                         $   !                     1   .   &   
       4   '      0   -                 /                   (   )   "          +                              %   *    
                <h1>Startup failed</h1>
                The startup of this notebook has failed. A known cause for this
                error is starting the notebook in Firefox in private mode.
                Please try again in a new Firefox window in normal mode. More
                background information about this problem can be found here:
                <br>
                <a href="https://jupyterlite.readthedocs.io/en/latest/howto/configure/advanced/service-worker.html" target="_blank">
                https://jupyterlite.readthedocs.io/en/latest/howto/configure/advanced/service-worker.html</a>
                 
            <h1>Inference</h1>
            In this area the trained artificial neuron can be tested:
            <ol>
                <li>
                    Change the selection of the two input checkboxes on the
                    left hand side.
                </li>
                <li>Watch the result on the righthand side.</li>
            </ol>
             
            <h1>Training</h1>
            In this area the artificial neuron can be trained:
            <ol>
                <li>
                    Select the checkboxes for the combinations you consider a
                    nice day in the rightmost column.
                </li>
                <li>Press the start button to start the learning process.</li>
                <li>
                    Press the same button to continue the learning process
                    step-by-step.
                </li>
                <li>
                    Press the "Finalize Epoch" button to continue learning
                    until the current epoch is finished (all inputs have been
                    tested once).
                </li>
                <li>
                    Press the "Reset" button to reset the learning process.
                </li>
            </ol>
             
            <h2>Resource consumption in machine learning</h2>
            <p>When training the artificial neuron, you may have noticed that
            this process can take very different lengths of time, depending on
            the randomly selected starting values of the weights. In the worst
            case, even in our simple example, the training can take several
            dozen weight adjustments before the artificial neuron produces the
            desired output. This hopefully makes it clear why the resource
            consumption for training an entire neural network with not just
            three but several billion parameters takes so long and consumes so
            many resources.</p>
             
            <h2>Task</h2>
            <p>In our simple example, there are 16 different variants, which
            could be a nice day (4 checkboxes = 4 bits, 2<sup>4</sup>=16).
            However, a single artificial neuron can <em>NOT</em> successfully
            learn two specific variants (the reason for this will follow). Try
            to find out which two variants these are.</p>
             
            <p>
            <ul>
                <li>
                    the dendrites become the inputs i<sub>1</sub> to
                    i<sub>n</sub>
                </li>
                <li>
                    the proximity of the dendrites to the axon is indicated by
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
             
            <p>A further consideration is that inputs that have delivered the
            strongest signal in the event of an error have also contributed the
            most to the error. The consequence of this is that the strength of
            the respective input should also be included in the correction:</p>
             
            <p>And now we have all the ingredients for a first small artificial
            neuron!</p>
             
            <p>And now you can perhaps already guess why powerful graphics
            cards are so popular for training and using AI models: Calculations
            with vectors, scalar products (and later also calculations with
            matrices) are precisely the use cases in which graphics cards are
            clearly superior to a normal processor, as they are built and
            optimized precisely for such calculations.</p>
             
            <p>As you can easily imagine, the answer to this question is very
            individual. Let's assume we are looking at the situation for a
            person for whom it is always a good day when it is weekend or
            their parents are visiting, or both:</p>
             
            <p>But how do you find the right values in a targeted and efficient
            way? Or to put it another way: How does an artificial neuron
            "learn"?</p>
             
            <p>But how does an artificial neuron "learn"? Let's try it with a
            very simple example...</p>
             
            <p>However, the difference may be so large that we overshoot the
            target with the new weights. We therefore introduce another
            factor, the learning rate &alpha;:</p>
             
            <p>If &delta; is positive, increase the weights, otherwise decrease
            them.</p>
             
            <p>If you have ever programmed before, you will most likely
            immediately see that we are dealing with an OR operation of the two
            inputs. But for once, we do <em>NOT</em> want to solve this task
            with explicit commands, but rather teach an artificial neuron to
            always generate the desired output for all possible variations of
            the inputs.</p>
             
            <p>In order to efficiently create solutions for these complex
            applications, the field of machine learning has established itself
            as a supplement to traditional programming. As is so often the
            case, the model for machine learning is taken from nature: the
            brain with its network of neurons. In order to understand the
            mathematical model based on this, let's first take a look at how a
            biological neuron works:</p>
             
            <p>In our example, there should be exactly two inputs, each of
            which can only accept truth values (i.e. "true" or "false"). The
            two inputs are:
            <ul>
                <li>It is weekend.</li>
                <li>Parents come to visit.</li>
            </ul>
            We want to use these two inputs to determine whether it is a good
            day:</p>
             
            <p>In simple terms, biological neurons function by receiving and
            transmitting electrical impulses. Reception takes place via the
            dendrites. The potentials of the electrical impulses received are
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
             
            <p>In the past, we typically used a programming language such as C,
            Java or Python to solve problems with computers, using explicit
            commands to create the data structures, algorithms, functions and
            ultimately executable programs required for the solution. This
            approach can be used to create excellent solutions in many areas.
            However, in complex situations, such as large amounts of data or
            dynamically changing environments, we reach the limits of
            conventional programming, e.g. in speech recognition, autonomous
            driving or medical diagnostics.</p>
             
            <p>Let us now return to point 3 of the above-mentioned learning
            process and the question of the
            <span style="color:red">reasonable direction</span> of the
            weighting adjustments. The answer to this question is as follows:
            Find the difference &delta; between the target value t (the value
            we expect given a certain input) and the output o (the value the
            neuron actually output):<p>
             
            <p>Looking at the last formula, it becomes obvious that we can now
            calculate with vectors:</p>
             
            <p>So that we do not always have to distinguish between the
            weights and the threshold value, we now simplify our model once
            again from a mathematical point of view: The threshold value
            becomes the special weight w<sub>0</sub>, whose input is always
            1 (also referred to as "bias" in the specialist literature):</p>
             
            <p>The almost 100 billion neurons in our human brain each have
            around 10,000 dendrites, enabling us to perform our outstanding
            learning and thinking abilities.</p>
             
            <p>The greater the error that occurred, the more the weights need
            to be corrected. The new value of each weight is therefore
            initially the product of its previous value and the size of the
            difference &delta; that occurred:</p>
             
            <p>The learning process for an artificial neuron could look like
            this, for example:
            <ol>
                <li>
                    Choose random values for the weights and the threshold.
                </li>
                <li>Determine and check the result for selected inputs.</li>
                <li>
                    If the result is not as expected, adjust the weights and
                    threshold in <span style="color:red">a reasonable
                    direction</span> and <span style="color:red">dimension
                    </span>. Jump back to point 2.
                </li>
                <li>Done, training successful.</li>
            </ol>
            </p>
             
            <p>The mathematically simplified model now looks like this:</p>
             
            <p>The simplified model of a biological neuron described above can
            now be converted into a mathematical model:</p>
             
            <p>Then we only have to set the
            <span style="color:red">weights</span> and the
            <span style="color:red">threshold value</span> in our artificial
            neuron accordingly:</p>
             
            <p>This obviously leads immediately to the next question: Which of
            the three weights should be adjusted and
            <span style="color:red">to what extent</span>?</p>
             
            <p>Three considerations are combined to answer this question:</p>
             
            <p>To do this, we must first assign numerical values to the truth
            values of the two inputs. Usually, "true" is assigned 1 and "false"
            0. Example: If it is weekend, the input i<sub>1</sub> is assigned
            the value 1 and if the parents are not coming to visit, the input
            i<sub>2</sub> is assigned the value 0.</p>
             <h1>Theory</h1> <h2>Artificial neuron</h2> <h2>Biological neuron</h2> <h2>Example</h2> <h2>Machine learning</h2> <h2>Why machine learning?</h2> Continue (adopt weights) Continue (calculate result) Continue (get random input) Epoch:  Finalize Epoch Learning Rate α: Nice Day? Reset Start (set random weights)) Visiting Parents Weekend artificial_neuron_en.png good_day_empty_en.png good_day_example_en.png learning_direction_and_dimension_en.png neuron_en.png simple_neuron_en.png Project-Id-Version: 0.1
Report-Msgid-Bugs-To: 
PO-Revision-Date: 2023-10-27 15:55+0200
Last-Translator: Ronny Standtke <ronny.standtke@gmx.net>
Language-Team: DE <DE@li.org>
Language: 
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
 
                <h1>Start fehlgeschlagen</h1>
                Der Start dieses Notebooks ist fehlgeschlagen. Eine bekannte                Ursache für diesen Fehler ist der Start des Notebooks in                Firefox im privaten Modus. Bitte versuchen Sie es noch einmal                in einem neuen Firefox-Fenster im normalen Modus. Weitere                Hintergrundinformationen zu diesem Problem sind hier zu                finden:                <br>
                <a href="https://jupyterlite.readthedocs.io/en/latest/howto/configure/advanced/service-worker.html" target="_blank">
                https://jupyterlite.readthedocs.io/en/latest/howto/configure/advanced/service-worker.html</a>
                 
            <h1>Inferenz</h1>
            In diesem Bereich kann die trainierte künstliche Nervenzelle getestet werden:
            <ol>
                <li>
                    Ändern Sie die Auswahl der beiden Eingabe-Checkboxen auf der linken Seite.                 </li>
                <li>Beobachten Sie das Ergebnis auf der rechten Seite.</li>
            </ol>
             
            <h1>Training</h1>
            In diesem Bereich kann die künstliche Nervenzelle trainiert werden:
            <ol>
                <li>
                    Wählen Sie in der rechten Spalte die Checkboxen für die Kombinationen, die Sie für schöne Tage halten.                 </li>
                <li>Drücken Sie den Startknopf, um den Lernprozess zu starten.</li>
                <li>Drücken Sie den gleichen Knopf, um den Lernprozess schrittweise fortzusetzen.                <li>
                    Drücken Sie den Knopf "Epoche abschließen", um so lange zu lernen, bis die aktuelle Epoche abgeschlossen wurde (alle Eingaben wurden einmal getestet).                </li>
                <li>
                    Drücken Sie den Knopf "Zurücksetzen", um den Lernprozess zurückzusetzen.                </li>
            </ol>
             
            <h2>Ressourcenverbrauch beim Maschinellen Lernen</h2>
            <p>Beim Training der künstlichen Nervenzelle ist Ihnen vielleicht aufgefallen, dass dieser Prozess, abhängig von den zufällig gewählten Startwerten der Gewichtungen, sehr unterschiedlich lang dauern kann. Im schlimmsten Fall kann das Training selbst in unserem einfachen Beispiel mehrere Dutzend Gewichtsanpassungen benötigen, bis die künstliche Nervenzelle die gewünschten Ausgaben produziert. Das verdeutlicht hoffentlich, warum der Ressourcenverbrauch für das Training eines ganzen neuronalen Netzwerks mit nicht nur drei sondern mit mehreren Milliarden Parametern so lange braucht und so enorm viel Ressourcen verbraucht.</p>
             
            <h2>Aufgabe</h2>
            <p>In unserem einfachen Beispiel gibt es 16 verschiedene Varianten, was ein schöner Tag sein könnte (4 Checkboxen = 4 Bits, 2<sup>4</sup>=16). Eine einzelne künstliche Nervenzelle kann jedoch zwei spezielle Varianten <em>NICHT</em> erfolgreich lernen (die Begründung dafür folgt noch). Versuchen Sie herauszufinden, welche zwei Varianten das sind.</p>
             
            <p>
            <ul>
                <li>
                    die Dendriten werden zu den Eingaben i<sub>1</sub> bis
                    i<sub>n</sub>
                </li>
                <li>
                    die Nähe der Dendriten zum Axon wird durch die                     jeweiligen Gewichtungen w<sub>1</sub> bis w<sub>n</sub>                     angegeben                </li>
                <li>
                    die Aufsummierung der empfangenen elektrischen Impulse im                     Zellkörper wird in der Übertragungsfunktion abgebildet                </li>
                <li>
                    das Aktionspotenzial wird im Schwellenwert &theta; hinterlegt                 </li>
                <li>
                    das Alles-oder-nichts-Prinzip wird durch die                     Aktivierungsfunktion abgebildet                </li>
            </ul>
            </p>
             
            <p>Eine weitere Überlegung ist, dass Eingaben, die im Fehlerfall das stärkste Signal geliefert haben, auch am meisten zum Fehler beigetragen haben. Die Konsequenz daraus ist, die Stärke der jeweiligen Eingabe ebenfalls in die Korrektur einzubeziehen:</p>
             
            <p>Und damit haben wir nun alle Zutaten für eine erste kleine künstliche Nervenzelle!</p>
             
            <p>Und jetzt ahnt man vielleicht schon, warum leistungsfähige Grafikkarten für das Training und die Anwendung von KI-Modellen so beliebt sind: Das Rechnen mit Vektoren, Skalarprodukten (und später auch noch Berechnungen mit Matrizen) sind genau die Anwendungsfälle, in denen Grafikkarten einem normalen Prozessor deutlich überlegen sind, da sie genau für solche Berechnungen gebaut und optimiert werden.</p>
             
            <p>Wie man sich leicht denken kann, ist die Antwort auf diese Frage sehr individuell. Nehmen wir an, wir betrachten die Situation für einen Menschen, für den immer dann ein guter Tag ist, wenn Wochenende ist oder die Eltern zu Besuch kommen, oder beides zutrifft:</p>
             
            <p>Aber wie findet man denn zielgerichtet und effizient die passenden Werte? Oder anders formuliert: Wie "lernt" eine künstliche Nervenzelle?</p>
             
            <p>Aber wie "lernt" jetzt so eine künstliche Nervenzelle? Versuchen wir es an einem ganz einfachen Beispiel...</p>
             
            <p>Die Differenz kann jedoch unter Umständen so groß sein, dass wir mit den neuen Gewichtungen viel zu weit über das Ziel hinausschießen. Daher führen wir einen weiteren Faktor, die Lernrate &alpha;, ein:</p>
             
            <p>Wenn &delta; positiv ist, erhöhe die Gewichtungen, ansonsten verringere sie.</p>
             
            <p>Wenn Sie schon einmal programmiert haben, werden Sie höchstwahrscheinlich sofort sehen, dass wir es hier mit einer ODER-Verknüpfung der beiden Eingaben zu tun haben. Aber wir wollen diese Aufgabe ausnahmsweise einmal <em>NICHT</em> mit expliziten Befehlen lösen, sondern einer künstlichen Nervenzelle beibringen, bei allen möglichen Variationen der Eingaben immer die gewünschte Ausgabe zu erzeugen.</p>
             
            <p>Um Lösungen für diese komplexen Anwendungsfälle effizient erstellen zu können, hat sich als Ergänzung zur klassischen Programmierung das Gebiet des maschinellen Lernens etabliert. Als Vorlage für das maschinelle Lernen dient, wie so häufig, das entsprechende Vorbild aus der Natur: das Gehirn mit seinem Verbund aus Nervenzellen. Um das darauf aufbauende mathematische Modell zu verstehen, schauen wir uns also zunächst die Funktionsweise einer biologischen Nervenzelle an:</p>
             
            <p>In unserem Beispiel soll es genau zwei Eingaben geben, die jeweils nur Wahrheitswerte (also "wahr" oder "falsch") annehmen können. Die beiden Eingaben sind:            <ul>
                <li>Es ist Wochenende.</li>
                <li>Die Eltern kommen zu Besuch.</li>
            </ul>
            Mit diesen beiden Eingaben wollen wir bestimmen, ob ein guter Tag ist:</p>
             
            <p>Biologische Nervenzellen funktionieren vereinfacht gesagt über den Empfang und die Weitergabe von elektrischen Impulsen. Der Empfang erfolgt über die Dendriten. Die Potenziale der empfangenen elektrischen Impulse werden im Zellkörper (Soma) aufsummiert. Sobald diese Summe einen bestimmten Schwellenwert, das Aktionspotenzial, überschreitet, wird der elektrische Impuls auf dem Axon weitergegeben. Das funktioniert nach dem Alles-oder-nichts-Prinzip: solange das Aktionspotenzial nicht erreicht ist, wird gar kein Impuls weitergegeben, sobald es erreicht ist, wird der Impuls weitergeleitet. Je näher sich ein Dendrit am Axon befindet, desto stärker ist der Einfluss seines eingehenden Impulses auf das Aktionspotenzial im Zellkörper. Seine Nähe zum Axon beeinflusst also die "Gewichtung" eines Dendrits. Weitergeleitete Impulse werden an den Axonterminalen auf andere Zellen (z.B. andere Nervenzellen, Drüsen- oder Muskelzellen) übertragen.</p>
             
            <p>Um Aufgaben mit Computern zu lösen, haben wir in der Vergangenheit typischerweise eine Programmiersprache wie z.B. C, Java oder Python verwendet, um mit expliziten Befehlen die zur Lösung notwendigen Datenstrukturen, Algorithmen, Funktionen und schlussendlich ausführbaren Programme zu erstellen. Mit dieser Herangehensweise lassen sich in vielen Bereichen hervorragende Lösungen erstellen. Dennoch stoßen wir in komplexen Situationen, wie bei großen Datenmengen oder sich dynamisch ändernden Umgebungen, an die Grenzen herkömmlicher Programmierung, z.B. bei Spracherkennung, autonomem Fahren oder medizinischen Diagnosen.</p>
             
            <p>Kommen wir nun zu Punkt 3 des oben erwähnten Lernprozesses und der Frage nach der <span style="color:red">sinnvollen Richtung</span> der Anpassung zurück. Die Antwort auf diese Frage ist wie folgt: Ermittle die Differenz &delta; zwischen dem Zielwert t (dem Wert, den wir bei einer bestimmten Eingabe erwarten) und der Ausgabe o (dem Wert, den die Nervenzelle tatsächlich ausgegeben hat):<p>
             
            <p>Beim Betrachten der letzten Formel wird offensichtlich, dass wir nun mit Vektoren rechnen können:</p>
             
            <p>Damit wir nicht immer zwischen den Gewichtungen und dem Schwellenwert unterscheiden müssen, vereinfachen wir unser Modell nun noch einmal aus mathematischer Sicht: Der Schwellenwert wird zur speziellen Gewichtung w<sub>0</sub>, dessen Eingabe immer 1 ist (in der Fachliteratur auch als "Bias" bezeichnet):</p>
             
            <p>Die knapp 100 Milliarden Nervenzellen in unserem menschlichen Gehirn verfügen jeweils über ungefähr 10.000 Dendriten und ermöglichen uns so unsere herausragenden Lern- und Denkfähigkeiten.</p>
             
            <p>Je stärker der aufgetretene Fehler war, umso stärker müssen die Gewichtungen korrigiert werden. Der neue Wert einer jeden Gewichtung ist also zunächst einmal das Produkt aus dessen bisherigem Wert und der Größe der aufgetretenen Differenz &delta;:</p>
             
            <p>Der Lernprozess bei einer künstlichen Nervenzelle könnte zum Beispiel so aussehen:             <ol>
                <li>
                    Wähle zufällige Werte für die Gewichtungen und den Schwellenwert.
                </li>
                <li>Ermittle und überprüfe das Ergebnis für ausgewählte Eingaben.</li>
                <li>
                    Falls das Ergebnis nicht den Erwartungen entspricht, passe die Gewichtungen und den Schwellenwert in <span style="color:red">eine sinnvolle Richtung</span> und <span style="color:red">in geeigneter Dimension</span> an. Springe zurück zu Punkt 2.
                </li>
                <li>Fertig, Training erfolgreich.</li>
            </ol>
            </p>
             
            <p>Das mathematisch vereinfachte Modell sieht nun so aus:</p>
             
            <p>Das beschriebene vereinfachte Modell einer biologischen Nervenzelle kann nun in ein mathematisches Modell überführt werden:</p>
             
            <p>Dann müssen wir eigentlich nur noch alle <span style="color:red">Gewichtungen</span> und den <span style="color:red">Schwellenwert</span> in unserer künstlichen Nervenzelle passend einstellen:</p>
             
            <p>Das führt offensichtlich sofort zur nächsten Frage: Welche der drei Gewichtungen sollte <span style="color:red">wie stark</span> angepasst werden?</p>
             
            <p>Zur Lösung dieser Frage werden drei Überlegungen miteinander kombiniert:</p>
             
            <p>Dazu müssen wir den Wahrheitswerten der beiden Eingänge zunächst numerische Werte zuordnen. Normalerweise wird "wahr" mit 1 und "falsch" mit 0 belegt. Beispiel: Wenn es Wochenende ist, erhält der Eingang i<sub>1</sub> den Wert 1 und wenn die Eltern nicht zu Besuch kommen, erhält der Eingang i<sub>2</sub> den Wert 0.</p>
             <h1>Theorie</h1> <h2>Künstliche Nervenzelle</h2> <h2>Biologische Nervenzelle</h2> <h2>Beispiel</h2> <h2>Maschinelles Lernen</h2> <h2>Warum maschinelles Lernen?</h2> Weiter (Gewichtungen anpassen) Weiter (Ausgabe berechnen) Weiter (zufälligen Input holen) Epoche:  Epoche abschließen Lernrate α Schöner Tag? Zurücksetzen Start (zufällige Gewichtungen setzen) Eltern zu Besuch Wochenende artificial_neuron_de.png good_day_empty_de.png good_day_example_de.png learning_direction_and_dimension_de.png neuron_de.png simple_neuron_de.png 