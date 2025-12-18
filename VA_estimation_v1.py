# 梯度下降估计va位置xyb
import json
import math
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import stft
from scipy.optimize import linear_sum_assignment
import time
import csv
import os


INF = 10**9
def Hungarian(cost_matrix):
    rows_idx, cols_idx = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[rows_idx, cols_idx].sum()
    #assign = [[x,y] for x,y in zip(rows_idx,cols_idx)]
    return cols_idx, cost

def MurtyPartition(N, a, type):
    """
     MurtyPartition partitioin node N with its minimum assignment a
     input:
      N - in Murty's original paper, N is a "node", i.e. a non empty
      subset of A, which contains all assignment schemes.
      a - a nMeas*1 vector containing one assignment scheme.
      type - type == 0 for N-to-N assignment problem, type == 1 for
          M-to-N assignment problem, where M > N, e.g. assign M jobs to
          N worker.
    Output:
      nodeList - containing the list of partition of N. The
          union of all assignments to all partitions and assignment {a}
          forms a complete set of assignments to N.
    """
    a = np.array(a).reshape(-1,1)
    nMeas = len(a)
    tmp = np.arange(nMeas).reshape(-1,1) #index col
    a = np.hstack([tmp,a])
    aset = {tuple(elem) for elem in iter(a.tolist())}
    inset = {tuple(elem) for elem in iter(N[0])}
    a1set = inset.intersection(aset)
    a2set = aset.difference(a1set)
    a2 = sorted(list(a2set),key=lambda x:x[0])
    nodelist = []
    length = len(a2set)-1 if type==0 else len(a2set)
    for i in range(length):
        if i == 0:
            Inclu = N[0]
        else:
            tmp = np.array([list(x) for x in a2[:i]])
            if N[0].size == 0:
                Inclu= tmp
            else:
                Inclu = np.vstack([N[0],tmp])
        tmp1 = np.array([list(a2[i])])
        if N[1].size == 0:
            Exclu = tmp1
        else:
            Exclu = np.vstack([N[1],tmp1])
        res = [Inclu,Exclu]
        nodelist.append(res)
    return nodelist

def murty(costMat, k):
    """
    Murty's algorithm finds out the kth minimum assignments, k = 1, 2, ...
    Syntax:
      solution = murty(costMat, k)
    In:
       costMat - nMeas*nTarg cost matrix.
       k - the command number controlling the output size.

    Out:
       solution - array containing the minimum, 2nd minimum, ...,
           kth minimum assignments and their costs. Each solution{i}
           contains {assgmt, cost} where assgmt is an nMeas*1 matrix
           giving the ith minimum assignment; cost is the cost of this
           assignment.
    """
    solution = [[] for _ in range(k)]
    t = 0
    assgmt, cost = Hungarian(costMat)
    solution[0] = [assgmt,cost]
    nodeRec = [np.array([]), np.array([])]
    assgmtRec = assgmt
    nodeList = MurtyPartition(nodeRec,assgmtRec,1)
    while t<k-1:
        minCost = INF #float("Inf")
        idxRec = -1
        #try to find one node in the nodeList with the minimum cost
        #print("length",len(nodeList))
        for i in range(len(nodeList)):
            node = nodeList[i].copy()
            Inclu = node[0]
            Exclu = node[1]
            mat = costMat.copy()
            #print(len(Inclu))
            for j in range(len(Inclu)):
                best = mat[Inclu[j,0],Inclu[j,1]]
                mat[Inclu[j,0],:] = INF
                mat[Inclu[j,0],Inclu[j,1]] = best
            for j in range(len(Exclu)):
                mat[Exclu[j,0],Exclu[j,1]] = INF
            assgmt,cost = Hungarian(mat)
            #print(assgmt)
            # if -1 in assgmt:
            #     continue
            if cost < minCost:
                minCost = cost
                nodeRec = node
                assgmtRec = assgmt
                idxRec = i
        if idxRec == -1:
            for i in range(t,k):
                solution[i] = solution[t].copy()
            t = k
            #print("adadad")
        else:
            t += 1
            solution[t] = [assgmtRec, minCost]
            lenNodeSet = set(range(len(nodeList)))
            idxSet = {idxRec}
            idx = lenNodeSet.difference(idxSet)
            nodetmp = [nodeList[i] for i in idx]
            nodeList = nodetmp + MurtyPartition(nodeRec, assgmtRec, 1)
    return solution

def eva2DLocalizationError(pastatesGT,pastatesEst):
    paCount=pastatesGT.shape[0]
    paEstCount=pastatesEst.shape[0]
    cost_matrix=np.sum((np.repeat(pastatesEst[:,0:2].reshape(paEstCount,1,2),paCount,axis=1)-np.repeat(pastatesGT.reshape(1,paCount,2),paEstCount,axis=0))**2,axis=2)
    solution=murty(cost_matrix,k=1)
    [_, sumse] = solution[0]
    rmse=np.sqrt(sumse/np.min((paCount,paEstCount)))
    return rmse


def Compute_EuclideanDist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def PowerScaling(powerlist):
    return powerlist

def MeasModel(pa,ue):
    return Compute_EuclideanDist(pa[0:2],ue)+pa[2]

def invert_mapping(mapping_m, n=None):
    """
    将 (m,) 的关联数组转换为 (n,) 的反映射数组。

    参数:
        mapping_m (np.ndarray): 形状为 (m,) 的数组。
                                值代表对应 m 索引映射到的 n 索引。
                                -1 代表无关联。
        n (int, optional):      输出数组的长度。
                                如果不提供，将根据 mapping_m 中的最大值自动推断。
                                建议显式提供 n 以确保输出形状符合预期。

    返回:
        np.ndarray: 形状为 (n,) 的数组。
                    值代表对应 n 索引映射回的 m 索引。
                    -1 代表无关联。
    """
    # 1. 确定输出数组的大小 n
    if n is None:
        # 如果未指定 n，则取输入中的最大索引值 + 1（需要排除 -1）
        max_val = np.max(mapping_m)
        if max_val == -1:
            n = 0 # 极端情况：全是 -1
        else:
            n = max_val + 1

    # 2. 初始化全为 -1 的输出数组
    # 使用整型 (int) 填充
    mapping_n = np.full(n, -1, dtype=mapping_m.dtype)

    # 3. 找到有效的关联（过滤掉 -1）
    # valid_mask 是一个布尔数组
    valid_mask = mapping_m != -1

    # 4. 获取有效关联的索引和值
    # sources 是 m 域中的索引 (0 到 m-1 中有效的部分)
    # targets 是 n 域中的索引 (即 mapping_m 中非 -1 的值)
    sources = np.where(valid_mask)[0]
    targets = mapping_m[valid_mask]

    # 5. 执行反映射赋值
    # 这一步利用了 NumPy 的花式索引，速度很快
    # 注意：如果 mapping_m 中有多个索引指向同一个 n (多对一)，
    # 后出现的索引会覆盖先出现的索引。
    if len(targets) > 0:
        mapping_n[targets.astype(np.int32)] = sources

    return mapping_n


def GetLogLikelihoodNeg(ueposGT,MeasMatrix,pastate):
    # 函数通过murty计算最佳匹配，返回所有时刻的最佳匹配与对应的总似然函数一阶近似
    # 先假设第一个测量值与antenna j0匹配，随后可以根据TDoA构造cost matrix并运行murty；随后遍历所有antenna，选择cost最小的天线与第一个测量值匹配
    # -Log_likelihood = sum(((EuclideanDist(Pos, pastate[j,0:2]) + pastate[j,2] - Meas[i]) - (EuclideanDist(Pos, pastate[j0,0:2]) + pastate[j0,2] - Meas[0]))**2), j = assign[i]

    timestamplength=ueposGT.shape[0]
    paCount=pastate.shape[0]
    EstMeasList=np.zeros((timestamplength,paCount))
    # get estmeaslest
    for step in range(timestamplength):
        uepos=ueposGT[step,:]
        for num in range(paCount):
            EstMeasList[step,num]=MeasModel(pastate[num,:],uepos)
    # Murty算association和似然函数
    LogLikelihoodList=np.zeros(timestamplength)
    AssignListM2E=[[] for _ in range(timestamplength)]
    AssignListE2M=[[] for _ in range(timestamplength)]
    for step in range(timestamplength):
        MeasDiff=MeasMatrix.copy()[step]            
        MeasDiff[:,0]=MeasDiff[:,0]-MeasDiff[0,0]    # 相对值
        MeasCount=len(MeasDiff)
        EstMeas=EstMeasList.copy()[step,:]        # 此处为绝对ToA
        loglikelihood_min=INF
        AssignMinM2E=np.zeros((MeasCount,1))
        if MeasCount==0:
            LogLikelihoodList[step]=0
            AssignListE2M[step] = -np.ones(paCount)
            continue
        elif MeasCount <= paCount:
            cost_matrix=np.zeros((MeasCount-1,paCount-1))
            for da1 in range(paCount):
                # 实际首达径指向第da1个基站：出发点是实际测量到的首达径几乎必定存在对应pa
                EstMeasDiff=EstMeas-EstMeas[da1]
                EstMeasDiff=np.delete(EstMeasDiff,da1)
                cost_matrix=np.square(np.repeat(MeasDiff[1:,0].reshape(MeasCount-1,1),paCount-1,axis=1)-np.repeat(EstMeasDiff.reshape(1,paCount-1),MeasCount-1,axis=0))
                # 加入power scaling
                cost_matrix_scaled=cost_matrix*np.repeat(PowerScaling(MeasDiff[1:,1]).reshape(MeasCount-1,1),paCount-1,axis=1)
                
                solution=murty(cost_matrix_scaled,k=1)
                [assignM2E, loglikelihood] = solution[0]
                assignM2E = assignM2E + (assignM2E >= da1)
                assignM2E = np.insert(assignM2E, 0, da1)
                if (loglikelihood < loglikelihood_min):
                    loglikelihood_min = loglikelihood.copy()
                    AssignMinM2E = assignM2E.copy()
        else: #简化版，只取能量最大的paCount个点做匹配，如果不这样做需要设计惩罚（正则）项
            cost_matrix=np.zeros((paCount-1,paCount-1))
            index=MeasDiff[:,1].argsort()[-paCount:]
            MeasDiffHighEner=MeasDiff.copy()[index,:]
            MeasDiffHighEner[:,0]=MeasDiffHighEner[:,0]-MeasDiffHighEner[0,0]    # 相对值
            for da1 in range(paCount):
                # 实际首达径指向第da1个基站：出发点是实际测量到的首达径几乎必定存在对应pa
                EstMeasDiff=EstMeas-EstMeas[da1]
                EstMeasDiff=np.delete(EstMeasDiff,da1)
                cost_matrix=np.square(np.repeat(MeasDiffHighEner[1:,0].reshape(paCount-1,1),paCount-1,axis=1)-np.repeat(EstMeasDiff.reshape(1,paCount-1),paCount-1,axis=0))
                # 加入power scaling
                cost_matrix_scaled=cost_matrix*np.repeat(PowerScaling(MeasDiffHighEner[1:,1]).reshape(paCount-1,1),paCount-1,axis=1)
                
                solution=murty(cost_matrix_scaled,k=1)
                [assignM2E, loglikelihood] = solution[0]
                assignM2E = assignM2E + (assignM2E >= da1)
                assignM2E = np.insert(assignM2E, 0, da1)
                if (loglikelihood < loglikelihood_min):
                    loglikelihood_min = loglikelihood.copy()
                    AssignMinM2E = -np.ones(MeasCount)
                    AssignMinM2E[index] = assignM2E.copy()
        AssignListM2E[step]=AssignMinM2E.astype(np.int32).copy()
        AssignListE2M[step]=invert_mapping(AssignMinM2E,paCount).astype(np.int32).copy()
        LogLikelihoodList[step]=loglikelihood_min.copy()
    return AssignListM2E,AssignListE2M,LogLikelihoodList


def GetGradient(ueposGT,MeasMatrix,pastatelist,AssignListM2E,AssignListE2M):
    timestamplength=ueposGT.shape[0]
    paCount=pastatelist.shape[0]
    EstMeasList=np.zeros((timestamplength,paCount))
    paStateGradient=np.zeros((paCount,3))
    LogLikelihoodList=np.zeros(timestamplength)
    # get estmeaslist
    for step in range(timestamplength):
        uepos=ueposGT[step,:]
        for num in range(paCount):
            EstMeasList[step,num]=MeasModel(pastatelist[num,:],uepos)

    for step in range(timestamplength):
        uepos=ueposGT[step,:]
        assignM2E=AssignListM2E[step]
        assignE2M=AssignListE2M[step]
        mask=(assignM2E != -1)
        validMeasCount=np.sum(mask)
        if np.any(mask):
            FirstArrival=np.argmax(mask)    # 第一个和state匹配的测量序号：首达径
        else:
            continue    # 没有匹配的测量
        MeasDiff=MeasMatrix.copy()[step]
        FirstArrivalpa=assignM2E[FirstArrival]  #首达径对应pa序号
        pastateFirstArrival=pastatelist[FirstArrivalpa,:]
        powerscale=PowerScaling(MeasDiff[:,1])
        
        MeasDiff[:,0]=MeasDiff[:,0]-MeasDiff[FirstArrival,0]    # 相对值
        EstMeas=EstMeasList.copy()[step,:]
        EstMeasDiff=EstMeas.copy()-EstMeas[FirstArrivalpa]
        for panum in range(paCount):
            if assignE2M[panum]==-1:
                continue
            residue=MeasDiff[assignE2M[panum],0]-EstMeasDiff[panum]
            stepGradient=np.zeros(3)
            stepGradientFirstArrival=np.zeros(3)
            pastate=pastatelist[panum,:]
            if panum==FirstArrivalpa:
                continue
            else:
                stepGradient[0:2]=-2*powerscale[assignE2M[panum]]*residue*(pastate[0:2]-uepos)/Compute_EuclideanDist(pastate[0:2],uepos)
                stepGradient[2]=-2*powerscale[assignE2M[panum]]*residue
                stepGradientFirstArrival[0:2]=2*powerscale[assignE2M[panum]]*residue*(pastateFirstArrival[0:2]-uepos)/Compute_EuclideanDist(pastateFirstArrival[0:2],uepos)
                stepGradientFirstArrival[2]=2*powerscale[assignE2M[panum]]*residue
            paStateGradient[panum,:]+=stepGradient
            paStateGradient[FirstArrivalpa,:]+=stepGradientFirstArrival
            LogLikelihoodList[step]+=powerscale[assignE2M[panum]]*residue**2
    
    return paStateGradient,LogLikelihoodList

if __name__ == '__main__':
    meterper512fft=3e8/512/120e3
    dir='./extractionresult/4_1/'
    data=np.load(dir+'preprocessresult.npz')
    # TDoAsGT,ToAsGT,detection_result,papos,uepos
    timestamplengthall=data['TDoAsGT'].shape[0]    #~1e4
    samplestep=1
    TDoAsGT=data['TDoAsGT'][0:timestamplengthall:samplestep,:]
    ToAsGT=data['ToAsGT'][0:timestamplengthall:samplestep,:]

    detection_result = data['detection_result'][0:timestamplengthall:samplestep, :, :].copy()
    eps = 1e-12
    detection_result[:, 1, :] = 10.0 * np.log10(np.clip(detection_result[:, 1, :], eps, None))

    ueposGT=data['uepos'][0:timestamplengthall:samplestep,:]
    paposGT=data['papos']
    timestamplength=TDoAsGT.shape[0]
    paCount=paposGT.shape[0]


    # ===== Step 0: 可视化输入数据（插到数据加载&切片之后即可） =====
    import matplotlib.pyplot as plt
    import numpy as np
    t = np.arange(timestamplength)
    n_bins = detection_result.shape[2]
    range_bins = np.arange(n_bins) * meterper512fft  # 每个bin对应的距离(米)
    # 0.1 UE轨迹 & PA位置
    plt.figure(figsize=(6, 6))
    plt.plot(ueposGT[:, 0], ueposGT[:, 1], lw=1.2, label="UE trajectory")
    plt.scatter(paposGT[:, 0], paposGT[:, 1], s=60, marker="^", label="PA (GT)")
    plt.axis("equal")
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("UE & PA geometry")
    plt.legend()
    # 0.2 detection_result 能量热力图（时间-距离bin）
    plt.figure(figsize=(11, 4))
    power_map = detection_result[:, 1, :].T  # (bin, time)
    plt.imshow(
        power_map,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], range_bins[0], range_bins[-1]],
    )
    plt.colorbar(label="Detection power")
    plt.xlabel("Time step")
    plt.ylabel("Range (m)")
    plt.title("detection_result power over time")
    # 0.3 每个时刻检测到的点数（便于看稀疏/密集程度）
    det_mask = detection_result[:, 0, :].astype(bool)  # (time, bin)
    # --- 2D：0/1 二值图（按 0101 的感觉画出来）---
    plt.figure(figsize=(11, 4))
    plt.imshow(
        det_mask.T.astype(int),   # (bin, time), 值为 0/1
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], range_bins[0], range_bins[-1]],
        interpolation="nearest",
    )
    plt.colorbar(label="Detection (0/1)")
    plt.xlabel("Time step")
    plt.ylabel("Range (m)")
    plt.title("Detection mask over time (0/1)")
    # --- 1D：每帧检测点数 ---
    plt.figure(figsize=(11, 3))
    det_count = det_mask.sum(axis=1)
    plt.plot(t, det_count)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time step")
    plt.ylabel("#detections")
    plt.title("Detections per step")
    # 0.5 GT ToA / TDoA 走势（画所有列）
    def _plot_all_cols(mat, title, show_legend=False):
        if mat is None:
            return
        if np.ndim(mat) == 1:
            plt.figure(figsize=(11, 3))
            plt.plot(t, mat)
            plt.grid(True, alpha=0.3)
            plt.title(title)
            plt.xlabel("Time step")
            plt.tight_layout()
            return
        if np.ndim(mat) == 2 and mat.shape[0] == timestamplength:
            plt.figure(figsize=(11, 3))
            for i in range(mat.shape[1]):  # 画所有列
                plt.plot(t, mat[:, i], lw=0.9, alpha=0.9, label=f"col {i}")
            plt.grid(True, alpha=0.3)
            plt.title(title)
            plt.xlabel("Time step")
            if show_legend and mat.shape[1] <= 20:  # 列太多就别画legend了
                plt.legend(ncol=5, fontsize=8)
            plt.tight_layout()
    _plot_all_cols(ToAsGT, "ToAsGT (all columns)", show_legend=False)
    _plot_all_cols(TDoAsGT, "TDoAsGT (all columns)", show_legend=False)

    plt.show()
    # ===== Step 0 end =====





    # Get tdoa measurenents from detection results
    MeasMatrix=[[] for _ in range(timestamplength)]
    for step in range(timestamplength):
        MeasMatrix[step]=np.stack([np.array(np.where(detection_result[step,0,:])[0].tolist())*meterper512fft,detection_result[step,1,np.where(detection_result[step,0,:])[0].tolist()]],axis=1)

    # Step 1: 初值选取
    initnoisevar=2.5e3 #xy上标准差50m
    pastateinit=np.concatenate([paposGT+np.random.normal(0,np.sqrt(initnoisevar/2),size=paposGT.shape),np.zeros((paposGT.shape[0],1))],axis=1)
    # pastateinit=pastateinit[0:5,:]

    # Step 2: 梯度下降
    # Step 2 (rewrite): 稳定版交替优化（关联刷新 + 梯度下降）
    # - 不改输入数据结构
    # - 仍用你现有的 GetLogLikelihoodNeg / GetGradient / eva2DLocalizationError
    # =========================
    iternum = 2000                   # 总迭代次数（参数更新次数）
    assignrefresh = 5                # 每 X 次迭代重新计算关联
    outer_loops = (iternum + assignrefresh - 1) // assignrefresh

    paEstIter = np.zeros((iternum + 1, paCount, 3), dtype=float)
    paEstIter[0, :, :] = pastateinit
    rmse2DiterList = np.zeros(iternum + 1, dtype=float)
    cost_iter = np.full(iternum + 1, np.nan, dtype=float)

    # --- 超参数（相对保守，先保证不发散）---
    base_lr = 1e-1                   # “初始”步长（线搜索会自动缩小）
    max_backtrack = 20               # 回溯线搜索最多缩小多少次
    shrink = 0.5                     # 每次回溯步长乘 shrink
    grad_clip = 1e6                  # 梯度裁剪阈值（按元素裁剪，防爆）
    min_lr = 1e-12                   # 最小步长，避免死循环
    verbose_every = 1                # 每几次打印一次

    counter = 0

    def _mean_cost(theta, A_m2e, A_e2m):
        """用 GetGradient 取 cost（保持与你现有实现一致）"""
        G, L = GetGradient(ueposGT, MeasMatrix, theta, A_m2e, A_e2m)
        return G, float(np.mean(L))


    from datetime import datetime
    log_dir = "iter_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "opt_log.txt")  # 固定文件名：每次都追加到同一个文件
    # 每次运行先追加一个分隔/时间戳，方便回看
    with open(log_path, "a") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"# RUN START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# iter  cost  rmse2D(m)  lr  dir  improve\n")



    for out in range(outer_loops):
        # 1) 关联刷新（在当前参数下）
        AssignListM2E, AssignListE2M, _ = GetLogLikelihoodNeg(
            ueposGT, MeasMatrix, paEstIter[counter, :, :]
        )

        # 2) 固定关联，做 assignrefresh 次更新（最后一轮可能不足）
        inner_steps = min(assignrefresh, iternum - counter)
        for _ in range(inner_steps):
            theta = paEstIter[counter, :, :]

            # 2.1 计算梯度与当前 cost
            Gradient, cur_cost = _mean_cost(theta, AssignListM2E, AssignListE2M)

            # ---- norm clip：保留梯度形状，只缩放整体 ----
            g_norm = np.linalg.norm(Gradient)
            g_max  = np.max(np.abs(Gradient))
            print(f"[dbg] grad_norm={g_norm:.3g}, max|grad|={g_max:.3g}")

            # 设一个“范数阈值”，别用 np.clip 按元素截平
            # 你现在原始 grad_norm ~ 7e8，先从 1e7 或 1e8 试
            g_clip = 1e7
            if g_norm > g_clip:
                scale = g_clip / (g_norm + 1e-12)
                Gradient = Gradient * scale
            else:
                scale = 1.0

            g_norm_c = np.linalg.norm(Gradient)
            g_max_c  = np.max(np.abs(Gradient))
            print(f"[dbg] (after norm-clip) grad_norm={g_norm_c:.3g}, max|grad|={g_max_c:.3g}, scale={scale:.3g}")

            # 用当前的 test_lr 估算一下“单步更新量”（还没考虑线搜索 shrink）
            test_lr = max(base_lr, 1e-6)
            step_est = (test_lr / max(timestamplength, 1)) * Gradient
            print(f"[dbg] step_est: mean|step|={np.mean(np.abs(step_est)):.3g}, max|step|={np.max(np.abs(step_est)):.3g}")
            print(f"[dbg] step_est XY max={np.max(np.abs(step_est[:, :2])):.3g}, bias max={np.max(np.abs(step_est[:, 2])):.3g}")



            # 2.2 梯度裁剪（防止一次更新跳太大）
            g_norm = np.linalg.norm(Gradient)
            if g_norm > grad_clip:
                Gradient = Gradient * (grad_clip / (g_norm + 1e-12))

            # 2.3 自动判定“往哪边走能让 cost 变好”
            # 说明：你原实现的 cost 可能是 LL(越大越好) 或 NLL(越小越好)
            # 我们用一次小试探决定方向，不猜符号
            test_lr = max(base_lr, 1e-6)
            step0 = (test_lr / max(timestamplength, 1)) * Gradient

            # 试探两个方向：theta - step0 vs theta + step0
            _, cost_minus = _mean_cost(theta - step0, AssignListM2E, AssignListE2M)
            _, cost_plus  = _mean_cost(theta + step0, AssignListM2E, AssignListE2M)

            # 选择“更好”的方向（更好=更小 or 更大？）
            # 我们直接选两者里更优的那个：看哪个让 cost 更接近“改善”
            # 但改善方向未知，所以用“相对变化更明显的那边”，再配合线搜索保证单调
            # 简化：默认目标是让 cost 下降（NLL）；若发现上升更好，则翻转目标
            # 通过比较两边与当前的差值决定
            d_minus = cost_minus - cur_cost
            d_plus  = cost_plus  - cur_cost

            # 如果 d_minus < d_plus，说明减方向更有利于“降低 cost”；反之加方向更有利
            # 同时判断“降低还是升高”为好：看两边哪边更优（幅度更大且朝同一方向）
            # 最稳：直接选让 |cost - cur_cost| 变化更大的那边作为“改善方向”，并在回溯中要求严格改善
            if abs(d_minus) >= abs(d_plus):
                direction = -1.0
                trial_cost = cost_minus
            else:
                direction = +1.0
                trial_cost = cost_plus

            # 判定“改善”到底是 cost 变小还是变大：看试探后更接近哪种
            # 若 trial_cost < cur_cost 说明“变小”更好，否则“变大”更好
            want_decrease = (trial_cost < cur_cost)

            # 2.4 回溯线搜索：找一个能“单调改善”的 lr
            lr = base_lr
            improved = False

            for bt in range(max_backtrack):
                step = (lr / max(timestamplength, 1)) * Gradient
                theta_new = theta + direction * step

                _, new_cost = _mean_cost(theta_new, AssignListM2E, AssignListE2M)

                if want_decrease:
                    ok = (new_cost < cur_cost)
                else:
                    ok = (new_cost > cur_cost)

                if ok:
                    improved = True
                    break

                lr *= shrink
                if lr < min_lr:
                    break

            if not improved:
                # 线搜索没找到改善步长：本次不更新（或极小更新），避免发散
                theta_new = theta.copy()
                new_cost = cur_cost

            # 2.5 记录 & 写回
            paEstIter[counter + 1, :, :] = theta_new

            # ===== 迭代可视化：PA估计 vs 真值 =====

            plot_every = 10
            save_dir = "iter_plots"
            os.makedirs(save_dir, exist_ok=True)

            if (counter % plot_every) == 0:
                est = paEstIter[counter + 1, :, :2]   # (paCount,2)
                gt  = paposGT[:, :2] if paposGT.shape[1] >= 2 else paposGT

                err = np.linalg.norm(est - gt, axis=1)  # 每个PA的2D误差

                fig = plt.figure(figsize=(6, 6))

                # 真值与估计点
                plt.scatter(gt[:, 0], gt[:, 1], marker='^', s=70, label='PA GT')
                plt.scatter(est[:, 0], est[:, 1], marker='o', s=40, label='PA Est')

                # 误差连线 + 误差数值标注
                for j in range(gt.shape[0]):
                    plt.plot([gt[j, 0], est[j, 0]], [gt[j, 1], est[j, 1]], linewidth=1)
                    plt.text(est[j, 0], est[j, 1], f"{err[j]:.1f}", fontsize=8)

                # UE 轨迹（可选）
                plt.plot(ueposGT[:, 0], ueposGT[:, 1], linewidth=1, alpha=0.6, label='UE traj')

                plt.axis('equal')
                plt.grid(True, alpha=0.3)
                plt.title(f"iter={counter} | rmse2D={rmse2DiterList[counter]:.3f} m")
                plt.xlabel("x (m)")
                plt.ylabel("y (m)")
                plt.legend()
                plt.tight_layout()

                # 保存到本地文件（不弹窗）
                out_path = os.path.join(save_dir, f"iter_{counter:05d}.png")
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)  # 关键：关闭图像，避免弹窗/内存堆积
            # ===== end =====



            cost_iter[counter] = cur_cost
            rmse2DiterList[counter] = eva2DLocalizationError(paposGT, theta)

            log_line = (
                f"{counter:6d}  "
                f"{cur_cost: .6e}  "
                f"{rmse2DiterList[counter]:7.3f}  "
                f"{lr:7.3g}  "
                f"{'+' if direction > 0 else '-'}  "
                f"{'down' if want_decrease else 'up'}\n"
            )

            if (counter % verbose_every) == 0:
                print(
                    f"iter={counter}, cost={cur_cost:.6g}, "
                    f"rmse2D={rmse2DiterList[counter]:.3f} m, "
                    f"lr={lr:.3g}, dir={'+' if direction>0 else '-'}, "
                    f"improve={'down' if want_decrease else 'up'}"
                )

            # 关键：同一个文件持续追加
            with open(log_path, "a") as f:
                f.write(log_line)

            counter += 1



    # 最后一个点也补一下 rmse（可选）
    rmse2DiterList[counter] = eva2DLocalizationError(paposGT, paEstIter[counter, :, :])
    # =========================
    # Step 2 end
    # =========================
