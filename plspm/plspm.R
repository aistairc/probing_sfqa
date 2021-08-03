library(plspm)
require(stringr)

sat_data <- read.csv('./csv/extended_bert_normalized.csv')
#sat_data <- read.csv('./csv/extended_bert_valid_normalized.csv')

dataset <- "FQ" # "FQ" "SQ" "WQ"
cause <- "glue_only" #"sent_only" "glue_only"

ED_block <- c(str_c(dataset, ".ED"))
EL_block <- c(str_c(dataset, ".EL"))
RP_block <- c(str_c(dataset, ".RP"))
QA_block <- c(str_c(dataset, ".Final"))

ED_scale <- rep("raw")
EL_scale <- rep("raw")
RP_scale <- rep("raw")
QA_scale <- rep("raw")

if (cause == "sent_only") {
  SUR <- c(0,0,0, 0,0,0, 0)
  SYN <- c(0,0,0, 0,0,0, 0)
  SEM <- c(0,0,0, 0,0,0, 0)

  ED <- c(1,1,1, 0,0,0, 0)
  EL <- c(0,0,0, 1,0,0, 0)
  RP <- c(1,1,1, 0,0,0, 0)

  QA <- c(0,0,0, 0,1,1, 0)

  sat_path <- rbind(SUR, SYN, SEM, ED, EL, RP, QA)

  SUR_block <- c("LEN", "WC")
  SYN_block <- c("DEP", "TC", "BS")
  SEM_block <- c("TEN", "SN", "ON", "OMO", "CI")

  sat_blocks <- list(SUR_block, SYN_block, SEM_block, ED_block, EL_block, RP_block, QA_block)

  SUR_scale <- rep("raw", 2)
  SYN_scale <- rep("raw", 3)
  SEM_scale <- rep("raw", 5)

  sat_scale <- list(SUR_scale, SYN_scale, SEM_scale, ED_scale, EL_scale, RP_scale, QA_scale)

  sat_modes <- c("A", "A", "A", "A", "A", "A", "A")

} else if (cause == "glue_only") {
  SST <- c(0,0,0, 0,0,0, 0)
  SPT <- c(0,0,0, 0,0,0, 0)
  IT <- c(0,0,0, 0,0,0, 0)

  ED <- c(1,1,1, 0,0,0, 0)
  EL <- c(0,0,0, 1,0,0, 0)
  RP <- c(1,1,1, 0,0,0, 0)

  QA <- c(0,0,0, 0,1,1, 0)

  sat_path <- rbind(SST, SPT, IT, ED, EL, RP, QA)

  SST_block <- c("COLA.EVAL.MCC", "SST.2.EVAL.ACC")
  SPT_block <- c("MRPC.EVAL.ACC", "MRPC.EVAL.F1", "STS.B.EVAL.PEARSON", "STS.B.EVAL.SPEARMANR") #, "QQP.EVAL.ACC", "QQP.EVAL.F1")
  IT_block <- c("MNLI.EVAL.MNLI.MM.ACC", "MNLI.EVAL.MNLI.ACC", "QNLI.EVAL.ACC", "RTE.EVAL.ACC") #, "WNLI.EVAL.ACC")

  sat_blocks <- list(SST_block, SPT_block, IT_block, ED_block, EL_block, RP_block, QA_block)

  SST_scale <- rep("raw", 2)
  SPT_scale <- rep("raw", 4)
  IT_scale <- rep("raw", 4)

  sat_scale <- list(SST_scale, SPT_scale, IT_scale, ED_scale, EL_scale, RP_scale, QA_scale)

  sat_modes <- c("A", "A", "A", "A", "A", "A", "A")

}

colnames(sat_path) <- rownames(sat_path)

satpls <- plspm(sat_data, sat_path, sat_blocks, scaling=sat_scale, modes=sat_modes, boot.val = TRUE, br = 600)

log <- summary(satpls)
capture.output(log, file=str_c('./', dataset, '-', cause, '.log'))
png(str_c('./', dataset, '-', cause, '.png'))
pairs(satpls$scores, panel=panel.smooth)
dev.off()
