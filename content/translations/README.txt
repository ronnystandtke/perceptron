Initial setup:
==============

1) Extract messages from Python script:
xgettext ../Perceptron.py

2) Edit comment at the top of the po file, e.g. with vi:
vi messages.po

3) Create language specific copies, e.g. for German:
cp messages.po de.po
cp messages.po de_CH.po

4) Edit language specific .po file (add header information and translations), e.g. with vi for German:
vi de.po
vi de_CH.po

5) Create directory to hold the .mo files, e.g. for German:
mkdir -p de/LC_MESSAGES
mkdir -p de_CH/LC_MESSAGES

6) Create .mo file for certain language, e.g. for German:
msgfmt -o de/LC_MESSAGES/perceptron.mo de.po
msgfmt -o de_CH/LC_MESSAGES/perceptron.mo de_CH.po


Update:
=======

1) Extract messages from Python script:
xgettext ../Perceptron.py

2) Update language specific po file, e.g. for German:
msgmerge -U de.po messages.po
msgmerge -U de_CH.po messages.po
vi de.po
vi de_CH.po

3) Update .mo file for certain language, e.g. for German:
msgfmt -o de/LC_MESSAGES/perceptron.mo de.po
msgfmt -o de_CH/LC_MESSAGES/perceptron.mo de_CH.po
