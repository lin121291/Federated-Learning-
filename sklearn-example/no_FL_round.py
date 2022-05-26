from sklearn.linear_model import LogisticRegression
import no_FL

if __name__ == "__main__":

    n=5
    train_answer='sn '
    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",   #???
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    no_FL.set_initial_params(model)

    for i in range(n):
        tmp,modele=no_FL.train(model) 
        #更新模型參數還沒做到
        train_answer=train_answer+str(tmp)+' '


    f = open("output.txt", "a")
    f.writelines(train_answer)
    f.writelines("\n")
    f.close()# 關閉檔案