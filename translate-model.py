from ocrd_typegroups_classifier.typegroups_classifier import TypegroupsClassifier


translation = {
    'griechisch': 'greek',
    'hebräisch': 'hebrew',
    'kursiv': 'italic',
    'andere_schrift': 'other_font',
    'nicht_schrift': 'not_a_font'
}


tgc = TypegroupsClassifier.load('ocrd_typegroups_classifier/models/classifier.tgc')
# backup
tgc.save('ocrd_typegroups_classifier/models/classifier-german-speaking.tgc')

print(tgc.classMap)
tgc.classMap.translate(translation)
print(tgc.classMap)
tgc.save('ocrd_typegroups_classifier/models/classifier.tgc')
