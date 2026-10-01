def SNI_2digits_to_sections(sni_code):
    """Return a dict mapping 2-digit SNI division codes to their letter section.

    Parameters
    ----------
    sni_code : str
        The SNI version to use. One of ``"SNI92"``, ``"SNI02"``, or ``"SNI07"``.

    Returns
    -------
    dict[str, str]
        Mapping from 2-digit division string (e.g. ``"85"``) to section letter
        (e.g. ``"N"``). The key ``"0"`` maps to ``"Unknown"`` for unclassified
        records.

    Notes
    -----
    Section boundaries differ across versions. SNI92 and SNI02 cover sections
    A–Q (17 sections); SNI07 covers A–U (21 sections) following the revised
    NACE Rev. 2 structure.

    Sources
    -------
    - SCB (1992) "SNI 1992 – Svensk näringsgrensindelning" [A–Q]
    - SCB (2002) "SNI 2002 – Svensk näringsgrensindelning" [A–Q]
    - SCB (2007) "SNI 2007 – Standard för svensk näringsgrensindelning" [A–U]
    """
    if sni_code == "SNI92":
        map_1992 = {
            "0": "Unknown",
            # A. Jordbruk, jakt och skogsbruk
            "01": "A", "02": "A",
            # B. Fiske
            "05": "B",
            # C. Utvinning av mineral
            "10": "C", "11": "C", "12": "C", "13": "C", "14": "C",
            # D. Tillverkning
            "15": "D", "16": "D", "17": "D", "18": "D", "19": "D",
            "20": "D", "21": "D", "22": "D", "23": "D", "24": "D",
            "25": "D", "26": "D", "27": "D", "28": "D", "29": "D",
            "30": "D", "31": "D", "32": "D", "33": "D", "34": "D",
            "35": "D", "36": "D", "37": "D",
            # E. El-, gas-, värme- och vattenförsörjning
            "40": "E", "41": "E",
            # F. Byggverksamhet
            "45": "F",
            # G. Partihandel och detaljhandel; reparation av fordon m.m.
            "50": "G", "51": "G", "52": "G",
            # H. Hotell- och restaurangverksamhet
            "55": "H",
            # I. Transport, magasinering och kommunikation
            "60": "I", "61": "I", "62": "I", "63": "I", "64": "I",
            # J. Finansiell verksamhet
            "65": "J", "66": "J", "67": "J",
            # K. Fastighets- och uthyrningsverksamhet; företagstjänster
            "70": "K", "71": "K", "72": "K", "73": "K", "74": "K",
            # L. Offentlig förvaltning och försvar; obligatorisk socialförsäkring
            "75": "L",
            # M. Utbildning
            "80": "M",
            # N. Hälso- och sjukvård, sociala tjänster
            "85": "N",
            # O. Andra samhälleliga och personliga tjänster
            "90": "O", "91": "O", "92": "O", "93": "O",
            # P. Hushållens verksamhet som arbetsgivare
            "95": "P",
            # Q. Internationella organisationer, ambassader m.m.
            "99": "Q",
        }
        return map_1992

    if sni_code == "SNI02":
        map_2002 = {
            "0": "Unknown",
            # A. Jordbruk, jakt och skogsbruk
            "01": "A", "02": "A",
            # B. Fiske
            "05": "B",
            # C. Utvinning av mineral
            "10": "C", "11": "C", "12": "C", "13": "C", "14": "C",
            # D. Tillverkning
            "15": "D", "16": "D", "17": "D", "18": "D", "19": "D",
            "20": "D", "21": "D", "22": "D", "23": "D", "24": "D",
            "25": "D", "26": "D", "27": "D", "28": "D", "29": "D",
            "30": "D", "31": "D", "32": "D", "33": "D", "34": "D",
            "35": "D", "36": "D", "37": "D",
            # E. El-, gas-, värme- och vattenförsörjning
            "40": "E", "41": "E",
            # F. Byggverksamhet
            "45": "F",
            # G. Partihandel och detaljhandel; reparation av fordon m.m.
            "50": "G", "51": "G", "52": "G",
            # H. Hotell- och restaurangverksamhet
            "55": "H",
            # I. Transport, magasinering och kommunikation
            "60": "I", "61": "I", "62": "I", "63": "I", "64": "I",
            # J. Finansiell verksamhet
            "65": "J", "66": "J", "67": "J",
            # K. Fastighets- och uthyrningsverksamhet; företagstjänster
            "70": "K", "71": "K", "72": "K", "73": "K", "74": "K",
            # L. Offentlig förvaltning och försvar; obligatorisk socialförsäkring
            "75": "L",
            # M. Utbildning
            "80": "M",
            # N. Hälso- och sjukvård, sociala tjänster
            "85": "N",
            # O. Andra samhälleliga och personliga tjänster
            "90": "O", "91": "O", "92": "O", "93": "O",
            # P. Hushållens verksamhet som arbetsgivare
            "95": "P",
            # Q. Internationella organisationer, ambassader m.m.
            "99": "Q",
        }
        return map_2002

    if sni_code == "SNI07":
        map_2007 = {
            "0": "Unknown",
            # A. Jordbruk, skogsbruk och fiske
            "01": "A", "02": "A", "03": "A",
            # B. Utvinning av mineral
            "05": "B", "06": "B", "07": "B", "08": "B", "09": "B",
            # C. Tillverkning
            "10": "C", "11": "C", "12": "C", "13": "C", "14": "C",
            "15": "C", "16": "C", "17": "C", "18": "C", "19": "C",
            "20": "C", "21": "C", "22": "C", "23": "C", "24": "C",
            "25": "C", "26": "C", "27": "C", "28": "C", "29": "C",
            "30": "C", "31": "C", "32": "C", "33": "C",
            # D. Försörjning av el, gas, värme och kyla
            "35": "D",
            # E. Vattenförsörjning; avloppsrening, avfallshantering och sanering
            "36": "E", "37": "E", "38": "E", "39": "E",
            # F. Byggverksamhet
            "41": "F", "42": "F", "43": "F",
            # G. Handel; reparation av motorfordon och motorcyklar
            "45": "G", "46": "G", "47": "G",
            # H. Transport och magasinering
            "49": "H", "50": "H", "51": "H", "52": "H", "53": "H",
            # I. Hotell- och restaurangverksamhet
            "55": "I", "56": "I",
            # J. Informations- och kommunikationsverksamhet
            "58": "J", "59": "J", "60": "J", "61": "J", "62": "J", "63": "J",
            # K. Finans- och försäkringsverksamhet
            "64": "K", "65": "K", "66": "K",
            # L. Fastighetsverksamhet
            "68": "L",
            # M. Verksamhet inom juridik, ekonomi, vetenskap och teknik
            "69": "M", "70": "M", "71": "M", "72": "M", "73": "M", "74": "M", "75": "M",
            # N. Uthyrning, fastighetsservice, resetjänster och andra stödtjänster
            "77": "N", "78": "N", "79": "N", "80": "N", "81": "N", "82": "N",
            # O. Offentlig förvaltning och försvar; obligatorisk socialförsäkring
            "84": "O",
            # P. Utbildning
            "85": "P",
            # Q. Vård och omsorg; sociala tjänster
            "86": "Q", "87": "Q", "88": "Q",
            # R. Kultur, nöje och fritid
            "90": "R", "91": "R", "92": "R", "93": "R",
            # S. Annan serviceverksamhet
            "94": "S", "95": "S", "96": "S",
            # T. Förvärvsarbete i hushåll; hushållens produktion av diverse varor och tjänster för eget bruk
            "97": "T", "98": "T",
            # U. Verksamhet vid internationella organisationer, utländska ambassader o.d.
            "99": "U",
        }
        return map_2007
