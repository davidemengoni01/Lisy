import speech_recognition as sr

def speech_analyzer(shared, speech_event):

    # inizializza recognizer
    r = sr.Recognizer()

    # usa il microfono come sorgente
    with sr.Microphone() as source:
        print("Calibrazione rumore ambiente... aspetta 1 secondo")
        r.adjust_for_ambient_noise(source, duration=1)

        print("Parla (CTRL+C per uscire)\n")

        while shared["running"]:
            speech_event.wait()

            if shared["running"] == False:
                break

            print("In ascolto...")
            
            try:
                # ascolta audio
                audio = r.listen(source)

                # riconoscimento (Google Web Speech API)
                text = r.recognize_google(audio, language="it-IT")

                print("Hai detto:", text)

                print(text[0])
                shared["last_speech"] = text

                speech_event.clear() 

            except sr.UnknownValueError:
                print("Non ho capito...")

            except sr.RequestError as e:
                print("Errore API:", e)

            except KeyboardInterrupt:
                print("\nUscita...")
                break
        