;(load "./chipotle/intrinsics.lisp")
(in-package :chipotle)

(defparameter simd-architecture (make-hash-table))

(defun intrinsic-tmp (&optional name)
  (lisp (if name
	    (cintern (format nil "~a~a" name (gensym "")))
	    (cintern (format nil "xmm_~a" (gensym ""))))))

(defun scalar-tmp ()
  (lisp 
   (cintern (format nil "scalar~a" (gensym "")))))

(defmacro defoperator (name operator)
  `(defmacro ,name (&rest rest)
     (let* ((ret (pop rest)))
       (loop while rest do
	 (setf ret `(,',operator ,ret ,(pop rest))))
       ret)))

(defmacro mm_pow_ps (x y)
  `(exp_ps (_mm_mul_ps ,y (log_ps ,x))))

(defmacro mm256_pow_ps (x y)
  `(exp256_ps (_mm256_mul_ps ,y (log256_ps ,x))))

(defmacro mm_abs_ps (value)
  `(_mm_max_ps (_mm_sub_ps (_mm_setzero_ps) ,value) ,value))

(defmacro mm256_abs_ps(value)
  `(_mm256_max_ps (_mm256_sub_ps (_mm256_setzero_ps) ,value) ,value))


(defoperator add_sse _mm_add_ps)
(defoperator sub_sse _mm_sub_ps)
(defoperator mul_sse _mm_mul_ps)
(defoperator div_sse _mm_div_ps)
(defoperator min_sse _mm_min_ps)
(defoperator max_sse _mm_max_ps)
(defoperator or_sse  _mm_or_ps)
(defoperator xor_sse _mm_xor_ps)
(defoperator and_sse _mm_and_ps)
(defoperator andnot_sse _mm_andnot_ps)
(defoperator pow_sse mm_pow_ps)

(defoperator add_avx _mm256_add_ps)
(defoperator sub_avx _mm256_sub_ps)
(defoperator mul_avx _mm256_mul_ps)
(defoperator div_avx _mm256_div_ps)
(defoperator min_avx _mm256_min_ps)
(defoperator max_avx _mm256_max_ps)
(defoperator or_avx  _mm256_or_ps)
(defoperator xor_avx _mm256_xor_ps)
(defoperator and_avx _mm256_and_ps)
(defoperator andnot_avx _mm256_andnot_ps)
(defoperator pow_avx mm256_pow_ps)

(defmacro avx-eq  (a b) `(_mm256_cmp_ps ,a ,b '_CMP_EQ_UQ))
(defmacro avx-neq (a b) `(_mm256_cmp_ps ,a ,b '_CMP_NEQ_UQ))
(defmacro avx-le  (a b) `(_mm256_cmp_ps ,a ,b '_CMP_LE_UQ))
(defmacro avx-lt  (a b) `(_mm256_cmp_ps ,a ,b '_CMP_LT_UQ))
(defmacro avx-ge  (a b) `(_mm256_cmp_ps ,a ,b '_CMP_GE_UQ))
(defmacro avx-gt  (a b) `(_mm256_cmp_ps ,a ,b '_CMP_GT_UQ))

(defmacro intrinsify (arch &body body)
  (lisp
   (setf (gethash arch simd-architecture) t)
   (let ((numbers (make-hash-table))
	 (syms '()))
     (labels ((substitute-numbers (item)
		(if (and (listp item)
			 (not (and (eql (first item) 'float-type)
				   (floatp (second item)))))
		    (loop for i in item collect (substitute-numbers i))
		    (if (or (floatp item)
			    (listp item))
			(let ((tmp-number nil)
			      (item (if (listp item)
					(second item)
					item)))
			  (if (not (gethash item numbers))
			      (setf (gethash item numbers) (cintern (format nil "xmm_constant_~a__~a" item (gensym "")))))
			  (setf tmp-number (gethash item numbers))
			  (setf (get tmp-number 'value) item)
			  (push tmp-number syms)
			  tmp-number)
			item))))

       (let ((body (substitute-numbers body))
	     (add (if (eq arch :sse) 'add_sse 'add_avx))
	     (sub (if (eq arch :sse) 'sub_sse 'sub_avx))
	     (div (if (eq arch :sse) 'div_sse 'div_avx))
	     (mul (if (eq arch :sse) 'mul_sse 'mul_avx))
	     (min (if (eq arch :sse) 'min_sse 'min_avx))
	     (max (if (eq arch :sse) 'max_sse 'max_avx))
	     (and (if (eq arch :sse) 'and_sse 'and_avx))
	     (or  (if (eq arch :sse) 'or_sse  'or_avx))
	     (xor (if (eq arch :sse) 'xor_sse 'xor_avx))
	     (andnot (if (eq arch :sse) 'andnot_sse 'andnot_avx))

	     (slli (if (eq arch :sse) '_mm_slli_epi32 '_mm256_slli_epi32))
	     (srli (if (eq arch :sse) '_mm_srli_epi32 '_mm256_srli_epi32)) 

	     (cmpgt (if (eq arch :sse) '_mm_cmpgt_ps 'avx-gt))
	     (cmpge (if (eq arch :sse) '_mm_cmpge_ps 'avx-ge))
	     (cmplt (if (eq arch :sse) '_mm_cmplt_ps 'avx-lt))
	     (cmple (if (eq arch :sse) '_mm_cmple_ps 'avx-le))
	     (cmpeq (if (eq arch :sse) '_mm_cmpeq_ps 'avx-eq))
	     (cmpneq (if (eq arch :sse) '_mm_cmpneq_ps 'avx-neq))

	     (abs (if (eq arch :sse) 'mm_abs_ps 'mm256_abs_ps))
	     (pow (if (eq arch :sse) 'pow_sse 'pow_avx))
	     (exp (if (eq arch :sse) 'exp_ps 'exp256_ps))

	     (or2 (if (eq arch :sse) '_mm_or_si128 '_mm256_or_si256))
	     (slli2 (if (eq arch :sse) '_mm_slli_si128 '_mm256_slli_si256))
	     (srli2 (if (eq arch :sse) '_mm_srli_si128 '_mm256_srli_si256))

	     (xmm-load  (if (eq arch :sse) '_mm_load_si128 '_mm256_load_si256))
	     (xmm-loadu (if (eq arch :sse) '_mm_loadu_si128 '_mm256_loadu_si256))
	     (xmm-store  (if (eq arch :sse) '_mm_store_si128 '_mm256_store_si256))
	     (xmm-storeu (if (eq arch :sse) '_mm_storeu_si128 '_mm256_storeu_si256))

	     (cvt-i2f (if (eq arch :sse) '_mm_cvtepi32_ps '_mm256_cvtepi32_ps))
	     (cvt-f2i (if (eq arch :sse) '_mm_cvtps_epi32 '_mm256_cvtps_epi32))
	     
	     (set-zero-float (if (eq arch :sse) '_mm_setzero_ps '_mm256_setzero_ps))
	     (set-float (if (eq arch :sse) '_mm_set1_ps '_mm256_set1_ps))

	     (xmm-type (if (eq arch :sse) '__m128 '__m256)))


	 ;(maphash #'(lambda (a b) (format t "nums: ~a, ~a~%" a b)) numbers)
	 ;(format t "BODY: ~s~%" body)
	  
	 `(cm
	   (macrolet
	       ((+ (&rest rest) `(,',add ,@rest))
		(- (&rest rest) `(,',sub ,@rest))
		(* (&rest rest) `(,',mul ,@rest))
		(/ (&rest rest) `(,',div ,@rest))

		(+= (&rest rest) `(set ,(first rest) (+ ,@rest)))
		(-= (&rest rest) `(set ,(first rest) (- ,@rest)))
		(*= (&rest rest) `(set ,(first rest) (* ,@rest)))
		(/= (&rest rest) `(set ,(first rest) (/ ,@rest)))

		(test (a b) `(funcall 'booja ,a ,b))
		(pow (&rest rest) `(,',pow ,@rest))
		(expf (&rest rest) `(,',exp ,@rest))
		(fminf (&rest rest) `(,',min ,@rest))
		(fmaxf (&rest rest) `(,',max ,@rest))
		(fabsf (value) `(,',abs ,value))

		(and (&rest rest) `(,',and ,@rest))
		(or  (&rest rest) `(,',or ,@rest))
		(xor (&rest rest) `(,',xor ,@rest))
		(andnot (&rest rest) `(,',andnot ,@rest))

		(>> (a b) `(,',srli ,a ,b))
		(<< (a b) `(,',slli ,a ,b))

		(>  (a b) `(,',cmpgt ,a ,b))
		(>= (a b) `(,',cmpge ,a ,b))
		(<  (a b) `(,',cmplt ,a ,b))
		(<= (a b) `(,',cmple ,a ,b))
		(== (a b) `(,',cmpeq ,a ,b))
		(!= (a b) `(,',cmpneq ,a ,b))

		(or2  (&rest rest) `(,',or2 ,@rest))
		(>>> (a b) `(,',srli2 ,a ,b))
		(<<< (a b) `(,',slli2 ,a ,b))


		(xmm-load   (a) `(,',xmm-load ,a))
		(xmm-loadu  (a) `(,',xmm-loadu ,a))
		(xmm-store  (a b) `(,',xmm-store ,a ,b))
		(xmm-storeu (a b) `(,',xmm-storeu ,a ,b))
		
		(cvt-i2f (a) `(,',cvt-i2f ,a))
		(cvt-f2i (a) `(,',cvt-f2i ,a))

		(set-f0 () `(,',set-zero-float))
		(set-fx (a) `(,',set-float ,a))

		(xmm-type (&optional extra) `(cintern (format nil "~a~a" ',',xmm-type (cl:if ',extra ',extra ""))))


		(float-type (exp) exp))
		;(float (exp) exp))

	     (decl ,(loop for i in (remove-duplicates syms) collect
			  `(const ,(if (eq arch :sse) '__m128 '__m256) ,i = (set-fx (cms-cuda::float-type ,(get i 'value)))))
			  ;`(const ,(if (eq arch :sse) '__m128 '__m256) ,i = (set-fx (cms-cuda::float ,(get i 'value)))))
			  ;; make float type
			  ;`(const ,(if (eq arch :sse) '__m128 '__m256) ,i = (set-fx ,(get i 'value))))
		   ,@body))))))))
