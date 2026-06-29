import threading
import speech_analyzer as sa
import hand_tracking as ht

def main():
    shared = {"running": True}

    speech_event = threading.Event()
    speech_event.set() 

    hand_thread = threading.Thread(target=ht.hand_tracking, args=(shared,speech_event))
    speech_thread = threading.Thread(target=sa.speech_analyzer, args=(shared, speech_event))

    hand_thread.start()
    speech_thread.start()

    hand_thread.join()
    speech_thread.join()


if __name__ == "__main__":
    main()
