from darenclair.ingest.arabe import est_renvoi, normaliser, recoller_mots
from darenclair.ingest.parse import split_articles
from darenclair.ingest.pdf import texte_span
from darenclair.retrieval.lexical import BM25, tokeniser


def car(c, x0, x1):
    return {"c": c, "bbox": [x0, 0, x1, 10]}


def test_ligature_lam_alef_inversee_reparee():
    # "الأولى" extrait "األولى" : l'alef de la ligature a une largeur nulle
    span = {"chars": [car("ا", 10, 12), car("أ", 8, 8), car("ل", 4, 8), car("و", 2, 4), car("ل", 1, 2), car("ى", 0, 1)]}
    assert texte_span(span) == "الأولى"
    # vrai article defini "ال" : l'alef a une largeur, on ne touche a rien
    vrai = {"chars": [car("ا", 10, 12), car("ل", 8, 10), car("ر", 5, 8)]}
    assert texte_span(vrai) == "الر"


def test_mots_coupes_recolles():
    base = " ".join(["على", "المادة", "البناء"] * 5)
    assert recoller_mots(base + " ع لى الب ناء ا لمادة") == base + " على البناء المادة"
    assert recoller_mots(base + " لى ع") == base + " على"  # morceaux inverses
    # deux vrais mots frequents ne sont pas colles
    deux = " ".join(["في", "البناء"] * 5)
    assert recoller_mots(deux) == deux


def test_chiffres_arabes_et_numero_sur_sa_ligne():
    assert normaliser("المادة ١٢٠ نص") == "المادة 120 نص"
    assert normaliser("نص.\nالمادة\n120\nالمكررة\nعنوان") == "نص.\nالمادة 120\nالمكررة\nعنوان"


def test_renvois_ecartes():
    assert est_renvoi("\nمن القانون رقم 12.90", 0)
    assert est_renvoi("\nأعلاه بالأسس", 0)
    assert not est_renvoi("\nالعناصر الخاضعة للرسم", 0)


TEXTE = """القسم الأول
قواعد الوعاء
الباب الأول
الرسوم المستحقة
المادة الأولى
أحكام عامة
يؤذن للجماعات الترابية باستيفاء الرسوم.
الفرع
الأول
الجماعات
المادة2
الرسوم المستحقة لفائدة الجماعات
تحدث الرسوم التالية كما في
المادة19
من القانون رقم 12.90
المادة 120
المكررة
توزيع العائد
يوزع عائد الرسم.
المادة168
مكرر مرتين
الأداء الإلكتروني
نص.
المادة 169
نسخت
الملحق
قائمة الأنشطة
"""


def test_decoupage_arabe():
    c = split_articles(TEXTE, "Loi 47-06", "fiscalite", "2021", slug="loi_47-06", langue="ar",
                       saut_renumerotation=0)
    assert [x.article for x in c] == ["1", "2", "120 bis", "168 ter"]  # 19 = renvoi, 169 abroge
    assert c[0].contexte == "القسم الأول : قواعد الوعاء > الباب الأول : الرسوم المستحقة"
    assert c[1].contexte.endswith("الفرع الأول : الجماعات")
    assert "المادة19 من القانون" in c[1].texte
    assert "قائمة الأنشطة" not in c[-1].texte  # annexe coupee
    assert c[0].langue == "ar" and c[0].texte_index().endswith(c[0].texte)
    assert "المادة 1" in c[0].texte_index()


def test_bm25_arabe():
    t = tokeniser("الرسوم المستحقة لفائدة الجماعات")
    assert tokeniser("للجماعات")[0] == tokeniser("الجماعات")[0]
    assert tokeniser("الأراضي")[0] == tokeniser("الاراضي")[0] == tokeniser("أراضي")[0]
    assert "في" not in t
    docs = {
        "tnb": "الرسم على الأراضي الحضرية غير المبنية",
        "th": "رسم السكن على العقارات المبنية",
        "fr": "La taxe sur les terrains urbains non batis",
    }
    bm = BM25(docs)
    assert bm.search("ما هو الرسم على الاراضي غير المبنية؟")[0] == "tnb"
    assert bm.search("terrains non batis")[0] == "fr"
