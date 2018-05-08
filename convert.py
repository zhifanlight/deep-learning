import os
import time

"""
change markdown syntax from MacDown to YouDaoYun Note 
    read from clipboard, process and paste to clipboard
"""

clip = os.popen('pbpaste').read()

clip = clip.replace(r'\\(', '`$').replace(r'\\)', '$`')
clip = clip.replace(r'\\', '\\').replace(r'\_', '_').replace(r'\*', '*').replace(r'\.', '.')
clip = clip.replace(r'\\newline', '\\\\')
clips = clip.split('\n')

ans = ''
start = False

for c in clips:
    if '$$' in c:
        if start:
            if c.strip() == '$$':
                ans += c.replace('$$', '```') + '\n'
                start = False
        else:
            if c.strip() == '$$':
                ans += c.replace('$$', '```math') + '\n'
                start = True
            else:
                left = 0
                right = len(c) - 1
                tabs = c.index('$$')
                while left < right and c[left] in (' ', '\t', '$'):
                    left += 1
                while left < right and c[right] in (' ', '\t', '$'):
                    right -= 1
                ans += '\t' * tabs + '```math' + '\n'
                ans += '\t' * tabs + c[left: right + 1] + '\n'
                ans += '\t' * tabs + '```' + '\n'
    else:
        ans += c + '\n'

name = str(time.time()).split('.')[0]
fp = open(name, 'w')
fp.write(ans)
fp.close()

os.system('pbcopy < %s' % name)
os.remove(name)
