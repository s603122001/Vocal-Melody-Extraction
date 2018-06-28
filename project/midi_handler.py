import mido
import numpy as np


def midi2score(song, subdivision, meter="notebased", melody_mark=False, melody_shift=0):
    '''
    Parse a midi file into a 2D score matrix.

    Parameters:
        melody_mark: bool
            If truth, the notes which are melody will be set to 2 in the score matrix.
        melody_shift: str
            Type of the target data.
    Returns:
        score_tmp: 2D array
            2D score matrix.
    '''
    mid = mido.MidiFile(song)

    length = mid.length

    bpm = 120

    # resolution
    if(meter == "timebased"):
        sec_per_tick = 0.02
    elif(meter == "notebased"):
        sec_per_tick = 60/bpm/subdivision

    channel_counter = 0
    voice_channel = -10
    drum_channel = -10

    for index, track in enumerate(mid.tracks):
        for msg in track:
            if(msg.type == "note_on"):
                if("Soprano" in track.name or "Voice" in track.name):
                    voice_channel = msg.channel

                elif("Drum" in track.name or "Drums" in track.name):
                    drum_channel = msg.channel
                channel_counter = max(channel_counter, msg.channel)
                break

    for msg in mid:
        if(msg.is_meta):
            if(msg.type == 'set_tempo'):
                tempo = msg.tempo
        else:
            if(msg.type == "note_on"):
                if (meter == "notebased"):
                    bpm = mido.tempo2bpm(tempo)
                    sec_per_tick = 60/bpm/subdivision
                    break

    channel_counter += 1
    score = np.zeros((channel_counter, int(length / sec_per_tick) + 1, 88))
    pos = 0
    time = 0
    for msg in mid:
        time += msg.time
        pos += int(np.round(msg.time / sec_per_tick))
        if(pos + 1 > score.shape[1]):
            score = np.append(score, np.zeros((channel_counter, pos - score.shape[1] + 1, 88)), axis=1)
        if(msg.is_meta):
            if(msg.type == 'set_tempo'):
                if(meter == "notebased"):
                    tempo = mido.tempo2bpm(msg.tempo)
                    sec_per_tick = 60/tempo/subdivision

        elif(msg.type == 'note_on'):
            p = msg.note - 21
            c = msg.channel

            if(c == drum_channel):
                pass
            else:
                if(msg.velocity == 0):
                    if (melody_mark and c == voice_channel):
                        score[c, pos:, p + (melody_shift * 12)] = 0
                    else:
                        score[c, pos:, p] = 0
                elif(msg.velocity != 1):
                    if(melody_mark and c == voice_channel):
                        score[c, pos:, p + (melody_shift * 12)] = 2
                    elif(melody_mark == False or c != voice_channel):
                        score[c, pos:, p] = 1

        elif(msg.type == 'note_off'):
            p = msg.note - 21
            c = msg.channel
            if(c == drum_channel):
                pass
            else:
                if(melody_mark and c == voice_channel):
                    score[c, pos:, p + (melody_shift * 12)] = 0
                else:
                    score[c, pos:, p] = 0

    score_tmp = np.sum(score, axis=0)
    score_tmp[score_tmp >= 1] = 1
    score_tmp += score[voice_channel]
    score_tmp[score_tmp >= 3] = 2
    return score_tmp