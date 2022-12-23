#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <fstream>

using namespace std;

int n, m;
vector<int> parent;
vector<vector<bool> > wall_r, wall_d;

int GetID(int i, int j) {
    return i * m + j;
}

int Find(int i) {
    if (parent[i] == i)  {
        return i;
    }
    return parent[i] = Find(parent[i]);
}

int Merge(int a, int b) {
    a = Find(a);
    b = Find(b);
    parent[a] = b;
}

void Input() {
    cin >> m >> n;
}

void Init() {
    srand(time(NULL));
    parent = vector<int>(n * m);
    wall_r = wall_d = vector<vector<bool> >(n, vector<bool>(m, true));
    for (int i = 0; i < n * m; i++) {
        parent[i] = i;
    }
}

void Eller() {
    // n = 세로, m = 가로
    for (int i = 0; i < n; i++) {
        // 인접한 방과 다른 집합에 속하면 50%의 확률로 벽을 부순다.
        for (int j = 0; j < m - 1; j++) {
            if (Find(GetID(i, j)) != Find(GetID(i, j + 1)) and (rand() & 1)) { // merge
                Merge(GetID(i, j + 1), GetID(i, j));
                wall_r[i][j] = false;
            }
        }
        if (i == n - 1) break;
        int l = 0, r = 0;
        while (l < m) {
            if (r + 1 >= m or Find(GetID(i, l)) != Find(GetID(i, r + 1))) {
                int size = r - l + 1;
                for (int wall_remove = 0; wall_remove < size; wall_remove++) {
                    int wall = l + rand() % size;
                    Merge(GetID(i + 1, wall), GetID(i, wall));
                    wall_d[i][wall] = false;
                }
                l = r + 1;
                r = r + 1;
            }
            else {
                r++;
            }
        }
    }

    // last line
    for (int j = 0; j < m - 1; j++) {
        if (Find(GetID(n - 1, j)) != Find(GetID(n - 1, j + 1))) {
            Merge(GetID(n - 1, j + 1), GetID(n - 1, j));
            wall_r[n - 1][j] = false;
        }
    }
}

void Write() {
    int draw_height = 2 * n + 1, draw_width = 2 * m + 1;
    vector<vector<char> > arr(draw_height, vector<char>(draw_width));

    // top and bottom
    for (int i = 0; i < draw_width; i++) {
        arr[0][i] = arr[draw_height - 1][i] = (i & 1) ? '-' : '+';
    }

    // left_end and right_end
    for (int i = 0; i < draw_height; i++) {
        arr[i][0] = arr[i][draw_width - 1] = (i & 1) ? '|' : '+';
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int cell_i = i * 2 + 1, cell_j = j * 2 + 1;
            arr[cell_i][cell_j] = ' ';
            arr[cell_i + 1][cell_j + 1] = '+';
            arr[cell_i][cell_j + 1] = wall_r[i][j] ? '|' : ' ';
            arr[cell_i + 1][cell_j] = wall_d[i][j] ? '-' : ' ';
        }
    }
	cout << "test" << endl;
    ofstream file_out("20181677.maz");
    for (auto &row : arr) {
        for (auto &ch : row) {
            file_out << ch;
        }
        file_out << '\n';
    }
    file_out.close();
}

int main() {
    Input();
    Init();
    Eller();
    Write();
    return 0;
}
