use crate::model::token::GemType;
use serde::{Deserialize, Serialize};

/// 5x5 版图的顺时针向外螺旋填充顺序坐标（0-indexed，共 25 格）
pub const SPIRAL_ORDER: [(usize, usize); 25] = [
    (2, 2), // 中心
    (3, 2), // 下
    (3, 1), // 左
    (2, 1), // 上
    (1, 1), // 上
    (1, 2), // 右
    (1, 3), // 右
    (2, 3), // 下
    (3, 3), // 下
    (4, 3), // 下
    (4, 2), // 左
    (4, 1), // 左
    (4, 0), // 左
    (3, 0), // 上
    (2, 0), // 上
    (1, 0), // 上
    (0, 0), // 上
    (0, 1), // 右
    (0, 2), // 右
    (0, 3), // 右
    (0, 4), // 右
    (1, 4), // 下
    (2, 4), // 下
    (3, 4), // 下
    (4, 4), // 终点
];

/// 连线候选（1 到 3 枚同方向相邻且非黄金标记）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineCandidate {
    pub count: u8,
    pub positions: [(usize, usize); 3],
    pub gems: [GemType; 3], // 仅前 count 个有效
}

/// 5x5 游戏棋盘
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Board {
    pub grid: [[Option<GemType>; 5]; 5],
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub const fn new() -> Self {
        Self {
            grid: [[None; 5]; 5],
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Option<GemType> {
        if r < 5 && c < 5 {
            self.grid[r][c]
        } else {
            None
        }
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, gem: Option<GemType>) {
        if r < 5 && c < 5 {
            self.grid[r][c] = gem;
        }
    }

    #[inline]
    pub fn take(&mut self, r: usize, c: usize) -> Option<GemType> {
        if r < 5 && c < 5 {
            self.grid[r][c].take()
        } else {
            None
        }
    }

    /// 统计棋盘上当前的标记总数
    pub fn count_tokens(&self) -> usize {
        let mut cnt = 0;
        for r in 0..5 {
            for c in 0..5 {
                if self.grid[r][c].is_some() {
                    cnt += 1;
                }
            }
        }
        cnt
    }

    /// 棋盘上是否存在至少 1 枚黄金
    pub fn has_gold(&self) -> bool {
        for r in 0..5 {
            for c in 0..5 {
                if matches!(self.grid[r][c], Some(GemType::Gold)) {
                    return true;
                }
            }
        }
        false
    }

    /// 沿官方 25 格螺旋轨道，从布袋列表（尾部弹出）中填满空格
    pub fn fill_spiral(&mut self, bag: &mut Vec<GemType>) {
        for &(r, c) in SPIRAL_ORDER.iter() {
            if self.grid[r][c].is_none() {
                if let Some(token) = bag.pop() {
                    self.grid[r][c] = Some(token);
                } else {
                    break;
                }
            }
        }
    }

    /// 检查棋盘是否有空格子
    pub fn has_empty_slot(&self) -> bool {
        for r in 0..5 {
            for c in 0..5 {
                if self.grid[r][c].is_none() {
                    return true;
                }
            }
        }
        false
    }

    /// 查找棋盘上所有合法的 1 至 3 枚连线相邻标记（绝不包含黄金与空格）
    pub fn find_all_lines(&self) -> Vec<LineCandidate> {
        let mut lines = Vec::with_capacity(128);

        // 1 枚标记（所有非黄金格子）
        for r in 0..5 {
            for c in 0..5 {
                if let Some(gem) = self.grid[r][c] {
                    if !gem.is_gold() {
                        lines.push(LineCandidate {
                            count: 1,
                            positions: [(r, c), (0, 0), (0, 0)],
                            gems: [gem, gem, gem],
                        });
                    }
                }
            }
        }

        // 4 个方向向量: 右 (0, 1), 下 (1, 0), 右下对角线 (1, 1), 左下对角线 (1, -1)
        const DIRS: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

        for &(dr, dc) in DIRS.iter() {
            for r in 0..5isize {
                for c in 0..5isize {
                    // 2 连线
                    let r1 = r + dr;
                    let c1 = c + dc;
                    if (0..5).contains(&r1) && (0..5).contains(&c1) {
                        if let (Some(g0), Some(g1)) = (
                            self.grid[r as usize][c as usize],
                            self.grid[r1 as usize][c1 as usize],
                        ) {
                            if !g0.is_gold() && !g1.is_gold() {
                                lines.push(LineCandidate {
                                    count: 2,
                                    positions: [
                                        (r as usize, c as usize),
                                        (r1 as usize, c1 as usize),
                                        (0, 0),
                                    ],
                                    gems: [g0, g1, g0],
                                });

                                // 3 连线
                                let r2 = r1 + dr;
                                let c2 = c1 + dc;
                                if (0..5).contains(&r2) && (0..5).contains(&c2) {
                                    if let Some(g2) = self.grid[r2 as usize][c2 as usize] {
                                        if !g2.is_gold() {
                                            lines.push(LineCandidate {
                                                count: 3,
                                                positions: [
                                                    (r as usize, c as usize),
                                                    (r1 as usize, c1 as usize),
                                                    (r2 as usize, c2 as usize),
                                                ],
                                                gems: [g0, g1, g2],
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        lines
    }
}
