(in-package :chipotle)

;; SSE intrinsics from sse_mathfun.h
(use-functions sin_ps cos_ps sincos_ps exp_ps log_ps)
;; AVX intrinsics from avx_mathfun.h
(use-functions sin256_ps cos256_ps sincos256_ps exp256_ps log256_ps)

(defmacro intrinsics# (arch &rest rest) 
  `(use-functions ,@(loop for i in rest collect 
		      (cintern (format nil "_mm~a_~a"
				       (cl:cond ((eq arch :sse) "")
						((eq arch :avx) "256"))
				       i)))))

(defmacro intrinsics (&rest rest)
 `(cl:progn
    (intrinsics# :sse ,@rest)
    (intrinsics# :avx ,@rest)))

(intrinsics

;; basic math
mul_ps
add_ps
sub_ps
div_ps 
sqrt_ps
max_ps
min_ps

;; set
set1_ps
setzero_ps
setzero_si128
setzero_si256

;; load, store
load_ps
loadu_ps
store_ps
storeu_ps

load_si128
loadu_si128
store_si128
storeu_si128

load_si256
loadu_si256
store_si256
storeu_si256

;; suffle
unpacklo_ps
unpacklo_epi8
unpacklo_epi16

unpackhi_ps
unpackhi_epi8
unpackhi_epi16

packs_epi32
packus_epi16

;; logic
and_ps
andnot_ps
or_ps
xor_ps

or_si128 ;;sse
or_si256 ;;avx

;; comparison
cmpeq_ps
cmpneq_ps
cmple_ps
cmplt_ps
cmpge_ps
cmpgt_ps

cmpeq_epi32

cmp_ps

;; conversion
cvtepi8_epi32
cvtepu8_epi32

cvtepi32_ps
cvtepu32_ps

cvtps_epi32
cvtsi128_si32

castsi128_ps
castsi256_ps

castps_si128
castps_si256

;; shift
srli_epi32
slli_epi32

srli_si128
slli_si128

srli_si256
slli_si256

)
