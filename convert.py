import os
import time

"""
change markdown syntax from Typora to YouDaoYun Note
    read from clipboard, process and paste to clipboard
"""

clip = os.popen('pbpaste').read()

clip = clip.replace(r'\newline', '\\\\')
clips = clip.split('\n')

ans = ''
equation_start = False
appear_code = False

for c in clips:
    if c.strip() == '$$':
        if equation_start:
            ans += c.replace('$$', '```') + '\n'
        else:
            ans += c.replace('$$', '```math') + '\n'
        equation_start = not equation_start
    elif '$$' in c:
        raise ValueError('$$ must exist in new line')
    elif '$' in c and not appear_code:
        appear_dollar = False
        for v in c:
            if v == '$':
                ans += '`$' if not appear_dollar else '$`'
                appear_dollar = not appear_dollar
            else:
                ans += v
        ans += '\n'
        if appear_dollar:
            print('[CHECK]: %s' % c)
    else:
        ans += c + '\n'
        if '```' in c:
            appear_code = not appear_code

name = str(time.time()).split('.')[0]
fp = open(name, 'w')
fp.write(ans)
fp.close()

os.system('pbcopy < %s' % name)
os.remove(name)
