from repr_checker.checker.check import ReprChecker

checker = ReprChecker()

document = '''	LONDON, England (CNN) -- Britain's accident prone
"Octopus UFO" is just one of hundreds of
unexplained sightings in the same area where a
wind turbine was wrecked over the weekend,
according to the latest reports. The Sun tabloid
newspaper's UFO splash. Britain's tabloid Sun
newspaper Thursday proclaimed from its front page
that a wind turbine was ruined after a UFO hit one
of its 20 meter-long blades in Conisholme,
Lincolnshire. It quoted residents who saw strange
balls of lights in the sky and heard a loud bang.
However, another British national newspaper said
the lights were just fireworks from a staff
member's dad's birthday celebration. Turbine
experts suggested it was a simple mechanical
failure. The plot thickened further Friday, with
The Sun saying it had been "bombarded" with
reports of UFO sightings from hundreds of
witnesses in the area where the turbine was
destroyed. Watch video on the UFO incident .
"There have been reports of flying saucers for
more than six months," the newspaper said. Local
John Harrison, 32, told The Sun he looked out of
his window and saw "a massive ball of light with
tentacles going right down to the ground." The
newspaper said "other respected witnesses, such as
local council chairman Robert Palmer and GP Jenny
Watson, described seeing 'streaking white
lights'." Quoting unnamed Ministry of Defence
"insiders," The Sun said the UFO sightings may be
an unmanned stealth bomber on test flights. It
said the Taranis "black delta-wing craft" was
being developed nearby to deliver bombs undetected
in war zones; back to the testing board then?
However, initial reports when the Taranis contract
was let last year said it would take at least four
years to develop with flight testing due 2010...
in Australia. CNN has also been "bombarded" with
messages, but mainly from people less than
convinced. J. Kale believed there was a very
simple explanation. "The octopus thing obviously
thought the wind turbine was a female doing a
mating dance and tried to mate with it.'''.replace('\n', ' ')
summary1 = '''British tabloid blames UFO for destroying wind
turbine blade . "Octopus UFO" may have been
unmanned stealth bomber on test flight, paper says
. CNN readers remain skeptical about role of UFO
in turbine's ruination .'''.replace('\n', ' ')

print(checker.check(document, summary1))

summary = '''"The octopus thing obviously
thought the wind turbine was a female doing a
mating dance and tried to mate with it.'''.replace('\n', ' ')

print('마지막 문장', checker.check(document, summary))

summary = '''"LONDON, England (CNN) -- Britain's accident prone
"Octopus UFO" is just one of hundreds of
unexplained sightings in the same area where a
wind turbine was wrecked over the weekend,
according to the latest reports.'''.replace('\n', ' ')

print('첫번째 문장', checker.check(document, summary))

summary = '''It quoted residents who saw strange
balls of lights in the sky and heard a loud bang.'''.replace('\n', ' ')

print('중간 문장', checker.check(document, summary))

summary = '''Watch video on the UFO incident .
"There have been reports of flying saucers for
more than six months," the newspaper said. However, initial reports when the Taranis contract
was let last year said it would take at least four
years to develop with flight testing due 2010.'''.replace('\n', ' ')

print('중간 문장 2개 mixed', checker.check(document, summary))

summary = '''UFO splash unnamed bombs was a female Mono was watching us. It six months said the turbine has also been believed explanations Jim but J. Kale flight LONDON, thickened 20 meter-long in Conisholme..'''.replace('\n', ' ')

print('단어 무작위 배열', checker.check(document, summary))

summary = '''The Sun said the UFO sightings may be
an unmanned stealth bomber on test flights with wind turbine.
Also they saw newspapers and reports but lights will be bombarded all the area in hundreds of UFO sightings.'''.replace('\n', ' ')

print('키워드 기반', checker.check(document, summary))

summary = '''Hello my name is UFO. When I got a new camera, I took a picture about UFO. So I show him to visit about them. Token indices sequence length is longer than the specified maximum sequence length for this model.'''

print('아무말 대잔치', checker.check(document, summary))

summary = '''Anarchists participated alongside the Bolsheviks in both February and October revolutions, and were initially enthusiastic about the Bolshevik coup. However, the Bolsheviks soon turned against the anarchists and other left-wing opposition, a conflict that culminated in the 1921 Kronstadt rebellion which the new government repressed.'''

print('리얼 아무말 대잔치', checker.check(document, summary))
