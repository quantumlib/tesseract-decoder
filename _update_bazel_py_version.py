# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

def main():
    version = sys.argv[1]
    lines = open('MODULE.bazel').read().splitlines()
    for i, l in enumerate(lines):
        if l.startswith('DEFAULT_PYTHON_VERSION = '):
            lines[i] = f'DEFAULT_PYTHON_VERSION = "{version}"'
            break
    with open('MODULE.bazel', 'w') as ouf:
        print(*lines, file=ouf, sep='\n')

if __name__ == '__main__':
    main()
