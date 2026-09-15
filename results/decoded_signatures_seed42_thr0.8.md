# Clause-Activation Signatures: iotm (seed 42, pi_thr=0.8)


## Family: Benign  (41 inculpatory + 0 exculpatory signature clauses)

(13/41 clauses drawn from Benign's own clause bank; 28 drawn from other classes' banks)

**Inculpatory (evidence FOR this family):**

- clause #21 (pi_hat=1.0, coef=+0.1204, positive, 7 literals):
  IF psh_flag_number>=bin2 AND NOT(Protocol Type>=bin3) AND NOT(syn_flag_number>=bin5) AND NOT(ack_flag_number>=bin6) AND NOT(ece_flag_number>=bin2) AND NOT(HTTPS>=bin1) AND NOT(IRC>=bin1)

- clause #87 (pi_hat=1.0, coef=+0.6322, positive, 4 literals):
  IF psh_flag_number>=bin1 AND ack_flag_number>=bin2 AND NOT(Rate>=bin1) AND NOT(ARP>=bin3)

- clause #91 (pi_hat=1.0, coef=+0.1773, positive, 4 literals):
  IF Header_Length>=bin3 AND Header_Length>=bin7 AND Rate>=bin3 AND NOT(psh_flag_number>=bin2)

- clause #132 (pi_hat=1.0, coef=+0.0562, positive, 5 literals):
  IF NOT(Header_Length>=bin7) AND NOT(Rate>=bin1) AND NOT(rst_count>=bin1) AND NOT(HTTPS>=bin1) AND NOT(DNS>=bin5)

- clause #178 (pi_hat=1.0, coef=+0.3641, positive, 85 literals):
  IF Header_Length>=bin2 AND Header_Length>=bin6 AND ack_flag_number>=bin1 AND ack_flag_number>=bin2 AND NOT(Header_Length>=bin9) AND NOT(Rate>=bin6) AND NOT(Rate>=bin8) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin2) AND NOT(fin_flag_number>=bin7) AND NOT(fin_flag_number>=bin9) AND NOT(syn_flag_number>=bin3) AND NOT(syn_flag_number>=bin5) AND NOT(syn_flag_number>=bin7) AND NOT(syn_flag_number>=bin8) AND NOT(syn_flag_number>=bin9) AND NOT(rst_flag_number>=bin1) AND NOT(rst_flag_number>=bin3) AND NOT(rst_flag_number>=bin5) AND NOT(rst_flag_number>=bin6) AND NOT(rst_flag_number>=bin7) AND NOT(rst_flag_number>=bin9) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin5) AND NOT(psh_flag_number>=bin6) AND NOT(psh_flag_number>=bin7) AND NOT(psh_flag_number>=bin8) AND NOT(ack_flag_number>=bin6) AND NOT(ack_flag_number>=bin7) AND NOT(ack_flag_number>=bin9) AND NOT(ece_flag_number>=bin1) AND NOT(ece_flag_number>=bin2) AND NOT(cwr_flag_number>=bin1) AND NOT(rst_count>=bin1) AND NOT(rst_count>=bin2) AND NOT(rst_count>=bin3) AND NOT(rst_count>=bin6) AND NOT(rst_count>=bin7) AND NOT(rst_count>=bin8) AND NOT(rst_count>=bin9) AND NOT(HTTP>=bin2) AND NOT(HTTP>=bin4) AND NOT(HTTP>=bin7) AND NOT(HTTP>=bin8) AND NOT(HTTP>=bin9) AND NOT(HTTPS>=bin1) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin4) AND NOT(HTTPS>=bin5) AND NOT(HTTPS>=bin8) AND NOT(HTTPS>=bin9) AND NOT(DNS>=bin1) AND NOT(DNS>=bin2) AND NOT(DNS>=bin4) AND NOT(DNS>=bin5) AND NOT(DNS>=bin6) AND NOT(DNS>=bin7) AND NOT(DNS>=bin8) AND NOT(DNS>=bin9) AND NOT(Telnet>=bin1) AND NOT(Telnet>=bin2) AND NOT(SSH>=bin2) AND NOT(SSH>=bin3) AND NOT(SSH>=bin4) AND NOT(DHCP>=bin1) AND NOT(DHCP>=bin2) AND NOT(DHCP>=bin3) AND NOT(DHCP>=bin4) AND NOT(DHCP>=bin5) AND NOT(DHCP>=bin6) AND NOT(DHCP>=bin7) AND NOT(DHCP>=bin9) AND NOT(ARP>=bin2) AND NOT(ARP>=bin3) AND NOT(ARP>=bin4) AND NOT(ARP>=bin6) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin5) AND NOT(ICMP>=bin6) AND NOT(ICMP>=bin9) AND NOT(IGMP>=bin1) AND NOT(IGMP>=bin4) AND NOT(IGMP>=bin5) AND NOT(IGMP>=bin7) AND NOT(IGMP>=bin8)

- clause #196 (pi_hat=1.0, coef=+0.3046, positive, 52 literals):
  IF Header_Length>=bin1 AND Header_Length>=bin2 AND Header_Length>=bin3 AND Header_Length>=bin4 AND Rate>=bin2 AND Rate>=bin3 AND ack_flag_number>=bin4 AND NOT(Header_Length>=bin8) AND NOT(Header_Length>=bin9) AND NOT(Protocol Type>=bin3) AND NOT(Rate>=bin4) AND NOT(Rate>=bin5) AND NOT(Rate>=bin6) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin5) AND NOT(fin_flag_number>=bin7) AND NOT(fin_flag_number>=bin8) AND NOT(fin_flag_number>=bin9) AND NOT(syn_flag_number>=bin3) AND NOT(syn_flag_number>=bin8) AND NOT(psh_flag_number>=bin3) AND NOT(psh_flag_number>=bin8) AND NOT(ack_flag_number>=bin5) AND NOT(ack_flag_number>=bin8) AND NOT(ece_flag_number>=bin2) AND NOT(rst_count>=bin2) AND NOT(rst_count>=bin3) AND NOT(HTTP>=bin1) AND NOT(HTTP>=bin2) AND NOT(HTTP>=bin4) AND NOT(HTTPS>=bin4) AND NOT(HTTPS>=bin7) AND NOT(DNS>=bin2) AND NOT(DNS>=bin3) AND NOT(DNS>=bin9) AND NOT(Telnet>=bin1) AND NOT(SMTP>=bin1) AND NOT(SSH>=bin2) AND NOT(SSH>=bin3) AND NOT(DHCP>=bin5) AND NOT(DHCP>=bin9) AND NOT(ARP>=bin3) AND NOT(ARP>=bin6) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin5) AND NOT(ICMP>=bin6) AND NOT(ICMP>=bin9) AND NOT(IGMP>=bin1) AND NOT(IGMP>=bin2) AND NOT(IGMP>=bin5) AND NOT(IGMP>=bin7) AND NOT(IGMP>=bin8)

- clause #290 (pi_hat=1.0, coef=+0.1712, negative, 34 literals):
  IF NOT(Header_Length>=bin5) AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin7) AND NOT(Rate>=bin4) AND NOT(Rate>=bin9) AND NOT(fin_flag_number>=bin3) AND NOT(fin_flag_number>=bin6) AND NOT(fin_flag_number>=bin7) AND NOT(syn_flag_number>=bin5) AND NOT(syn_flag_number>=bin7) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin6) AND NOT(psh_flag_number>=bin8) AND NOT(ack_flag_number>=bin3) AND NOT(ece_flag_number>=bin1) AND NOT(rst_count>=bin3) AND NOT(rst_count>=bin5) AND NOT(rst_count>=bin9) AND NOT(HTTP>=bin2) AND NOT(HTTP>=bin7) AND NOT(HTTPS>=bin1) AND NOT(HTTPS>=bin3) AND NOT(DNS>=bin4) AND NOT(DNS>=bin8) AND NOT(SSH>=bin3) AND NOT(DHCP>=bin5) AND NOT(DHCP>=bin6) AND NOT(DHCP>=bin8) AND NOT(DHCP>=bin9) AND NOT(ARP>=bin1) AND NOT(ARP>=bin3) AND NOT(ARP>=bin8) AND NOT(ICMP>=bin8) AND NOT(IGMP>=bin8)

- clause #345 (pi_hat=1.0, coef=+0.1550, negative, 121 literals):
  IF NOT(Header_Length>=bin2) AND NOT(Header_Length>=bin3) AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin5) AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin7) AND NOT(Header_Length>=bin8) AND NOT(Header_Length>=bin9) AND NOT(Protocol Type>=bin2) AND NOT(Protocol Type>=bin3) AND NOT(Rate>=bin5) AND NOT(Rate>=bin7) AND NOT(Rate>=bin8) AND NOT(Rate>=bin9) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin3) AND NOT(fin_flag_number>=bin4) AND NOT(fin_flag_number>=bin5) AND NOT(fin_flag_number>=bin6) AND NOT(fin_flag_number>=bin9) AND NOT(syn_flag_number>=bin2) AND NOT(syn_flag_number>=bin3) AND NOT(syn_flag_number>=bin4) AND NOT(syn_flag_number>=bin5) AND NOT(syn_flag_number>=bin6) AND NOT(syn_flag_number>=bin7) AND NOT(syn_flag_number>=bin9) AND NOT(rst_flag_number>=bin2) AND NOT(rst_flag_number>=bin3) AND NOT(rst_flag_number>=bin4) AND NOT(rst_flag_number>=bin5) AND NOT(rst_flag_number>=bin6) AND NOT(rst_flag_number>=bin7) AND NOT(psh_flag_number>=bin1) AND NOT(psh_flag_number>=bin2) AND NOT(psh_flag_number>=bin3) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin5) AND NOT(psh_flag_number>=bin7) AND NOT(psh_flag_number>=bin8) AND NOT(psh_flag_number>=bin9) AND NOT(ack_flag_number>=bin1) AND NOT(ack_flag_number>=bin3) AND NOT(ack_flag_number>=bin4) AND NOT(ack_flag_number>=bin5) AND NOT(ack_flag_number>=bin6) AND NOT(ack_flag_number>=bin7) AND NOT(ack_flag_number>=bin8) AND NOT(ack_flag_number>=bin9) AND NOT(ece_flag_number>=bin1) AND NOT(ece_flag_number>=bin2) AND NOT(cwr_flag_number>=bin1) AND NOT(cwr_flag_number>=bin2) AND NOT(rst_count>=bin1) AND NOT(rst_count>=bin4) AND NOT(rst_count>=bin5) AND NOT(rst_count>=bin7) AND NOT(rst_count>=bin9) AND NOT(HTTP>=bin1) AND NOT(HTTP>=bin3) AND NOT(HTTP>=bin5) AND NOT(HTTP>=bin6) AND NOT(HTTP>=bin7) AND NOT(HTTP>=bin8) AND NOT(HTTP>=bin9) AND NOT(HTTPS>=bin2) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin4) AND NOT(HTTPS>=bin5) AND NOT(HTTPS>=bin6) AND NOT(HTTPS>=bin7) AND NOT(HTTPS>=bin8) AND NOT(HTTPS>=bin9) AND NOT(DNS>=bin1) AND NOT(DNS>=bin2) AND NOT(DNS>=bin3) AND NOT(DNS>=bin4) AND NOT(DNS>=bin5) AND NOT(DNS>=bin6) AND NOT(DNS>=bin7) AND NOT(DNS>=bin8) AND NOT(DNS>=bin9) AND NOT(Telnet>=bin1) AND NOT(Telnet>=bin2) AND NOT(SMTP>=bin1) AND NOT(SMTP>=bin2) AND NOT(SSH>=bin1) AND NOT(SSH>=bin3) AND NOT(SSH>=bin4) AND NOT(SSH>=bin5) AND NOT(IRC>=bin1) AND NOT(IRC>=bin2) AND NOT(DHCP>=bin1) AND NOT(DHCP>=bin2) AND NOT(DHCP>=bin3) AND NOT(DHCP>=bin4) AND NOT(DHCP>=bin5) AND NOT(DHCP>=bin6) AND NOT(DHCP>=bin7) AND NOT(DHCP>=bin8) AND NOT(DHCP>=bin9) AND NOT(ARP>=bin1) AND NOT(ARP>=bin3) AND NOT(ARP>=bin4) AND NOT(ARP>=bin6) AND NOT(ICMP>=bin2) AND NOT(ICMP>=bin3) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin5) AND NOT(ICMP>=bin6) AND NOT(ICMP>=bin7) AND NOT(ICMP>=bin8) AND NOT(ICMP>=bin9) AND NOT(IGMP>=bin1) AND NOT(IGMP>=bin2) AND NOT(IGMP>=bin3) AND NOT(IGMP>=bin4) AND NOT(IGMP>=bin5) AND NOT(IGMP>=bin6) AND NOT(IGMP>=bin8) AND NOT(IGMP>=bin9)

- clause #357 (pi_hat=1.0, coef=+0.1442, negative, 24 literals):
  IF Rate>=bin1 AND ack_flag_number>=bin1 AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin7) AND NOT(syn_flag_number>=bin4) AND NOT(rst_flag_number>=bin4) AND NOT(psh_flag_number>=bin8) AND NOT(ece_flag_number>=bin3) AND NOT(cwr_flag_number>=bin1) AND NOT(rst_count>=bin9) AND NOT(HTTP>=bin1) AND NOT(HTTP>=bin4) AND NOT(HTTPS>=bin6) AND NOT(DNS>=bin7) AND NOT(DNS>=bin8) AND NOT(Telnet>=bin2) AND NOT(SSH>=bin5) AND NOT(DHCP>=bin2) AND NOT(DHCP>=bin4) AND NOT(ARP>=bin6) AND NOT(ICMP>=bin3) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin6) AND NOT(IGMP>=bin5)

- clause #401 (pi_hat=1.0, coef=+0.0570, positive, 1 literals)  [from DDoS's bank]:
  IF NOT(Rate>=bin4)

- clause #475 (pi_hat=1.0, coef=+0.0587, positive, 2 literals)  [from DDoS's bank]:
  IF NOT(Rate>=bin4) AND NOT(syn_flag_number>=bin8)

- clause #741 (pi_hat=1.0, coef=+0.0533, negative, 1 literals)  [from DDoS's bank]:
  IF psh_flag_number>=bin2

- clause #748 (pi_hat=1.0, coef=+0.0634, negative, 7 literals)  [from DDoS's bank]:
  IF NOT(Rate>=bin4) AND NOT(rst_flag_number>=bin9) AND NOT(psh_flag_number>=bin8) AND NOT(psh_flag_number>=bin9) AND NOT(HTTPS>=bin8) AND NOT(ARP>=bin4) AND NOT(IGMP>=bin4)

- clause #753 (pi_hat=1.0, coef=+0.0632, negative, 1 literals)  [from DDoS's bank]:
  IF NOT(Rate>=bin4)

- clause #798 (pi_hat=1.0, coef=+0.0647, negative, 1 literals)  [from DDoS's bank]:
  IF NOT(Rate>=bin4)

  ... and 26 more (see JSON for full list)


## Family: DDoS  (29 inculpatory + 0 exculpatory signature clauses)

(9/29 clauses drawn from DDoS's own clause bank; 20 drawn from other classes' banks)

**Inculpatory (evidence FOR this family):**

- clause #270 (pi_hat=1.0, coef=+0.0449, negative, 1 literals)  [from Benign's bank]:
  IF Rate>=bin6

- clause #312 (pi_hat=1.0, coef=+0.0515, negative, 1 literals)  [from Benign's bank]:
  IF Rate>=bin6

- clause #354 (pi_hat=1.0, coef=+0.6842, negative, 8 literals)  [from Benign's bank]:
  IF Rate>=bin3 AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin7) AND NOT(psh_flag_number>=bin1) AND NOT(psh_flag_number>=bin2) AND NOT(ack_flag_number>=bin2) AND NOT(ack_flag_number>=bin3)

- clause #431 (pi_hat=1.0, coef=+0.2324, positive, 5 literals):
  IF Rate>=bin4 AND Rate>=bin6 AND NOT(Header_Length>=bin3) AND NOT(syn_flag_number>=bin1) AND NOT(DNS>=bin6)

- clause #436 (pi_hat=1.0, coef=+0.4512, positive, 2 literals):
  IF Rate>=bin7 AND NOT(Header_Length>=bin4)

- clause #438 (pi_hat=1.0, coef=+0.1281, positive, 3 literals):
  IF Rate>=bin5 AND syn_flag_number>=bin1 AND rst_flag_number>=bin1

- clause #471 (pi_hat=1.0, coef=+0.4409, positive, 2 literals):
  IF Rate>=bin7 AND NOT(Header_Length>=bin4)

- clause #508 (pi_hat=1.0, coef=+0.2309, positive, 47 literals):
  IF Rate>=bin6 AND NOT(Header_Length>=bin1) AND NOT(Header_Length>=bin2) AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin7) AND NOT(Protocol Type>=bin2) AND NOT(Protocol Type>=bin3) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin7) AND NOT(fin_flag_number>=bin8) AND NOT(syn_flag_number>=bin2) AND NOT(syn_flag_number>=bin6) AND NOT(rst_flag_number>=bin6) AND NOT(psh_flag_number>=bin2) AND NOT(psh_flag_number>=bin3) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin7) AND NOT(psh_flag_number>=bin9) AND NOT(ack_flag_number>=bin4) AND NOT(ack_flag_number>=bin6) AND NOT(cwr_flag_number>=bin1) AND NOT(cwr_flag_number>=bin2) AND NOT(rst_count>=bin1) AND NOT(rst_count>=bin5) AND NOT(HTTP>=bin3) AND NOT(HTTP>=bin7) AND NOT(HTTP>=bin8) AND NOT(HTTPS>=bin2) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin7) AND NOT(DNS>=bin1) AND NOT(DNS>=bin4) AND NOT(DNS>=bin9) AND NOT(Telnet>=bin2) AND NOT(SMTP>=bin1) AND NOT(SMTP>=bin2) AND NOT(SSH>=bin4) AND NOT(DHCP>=bin1) AND NOT(ARP>=bin1) AND NOT(ARP>=bin2) AND NOT(ARP>=bin6) AND NOT(ARP>=bin7) AND NOT(ARP>=bin8) AND NOT(ARP>=bin9) AND NOT(ICMP>=bin2) AND NOT(ICMP>=bin3) AND NOT(IGMP>=bin1)

- clause #510 (pi_hat=1.0, coef=+0.2579, positive, 3 literals):
  IF Rate>=bin7 AND NOT(syn_flag_number>=bin2) AND NOT(rst_flag_number>=bin2)

- clause #530 (pi_hat=1.0, coef=+0.3092, positive, 66 literals):
  IF Header_Length>=bin1 AND Header_Length>=bin3 AND Protocol Type>=bin1 AND Rate>=bin2 AND Rate>=bin3 AND Rate>=bin6 AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin8) AND NOT(Header_Length>=bin9) AND NOT(Protocol Type>=bin3) AND NOT(Rate>=bin9) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin2) AND NOT(fin_flag_number>=bin3) AND NOT(fin_flag_number>=bin6) AND NOT(fin_flag_number>=bin9) AND NOT(syn_flag_number>=bin1) AND NOT(syn_flag_number>=bin2) AND NOT(syn_flag_number>=bin9) AND NOT(rst_flag_number>=bin1) AND NOT(rst_flag_number>=bin4) AND NOT(rst_flag_number>=bin5) AND NOT(psh_flag_number>=bin1) AND NOT(psh_flag_number>=bin2) AND NOT(psh_flag_number>=bin5) AND NOT(psh_flag_number>=bin6) AND NOT(psh_flag_number>=bin7) AND NOT(psh_flag_number>=bin9) AND NOT(ack_flag_number>=bin2) AND NOT(ack_flag_number>=bin4) AND NOT(ack_flag_number>=bin5) AND NOT(ece_flag_number>=bin1) AND NOT(ece_flag_number>=bin2) AND NOT(ece_flag_number>=bin3) AND NOT(rst_count>=bin4) AND NOT(rst_count>=bin7) AND NOT(rst_count>=bin8) AND NOT(rst_count>=bin9) AND NOT(HTTP>=bin7) AND NOT(HTTP>=bin8) AND NOT(HTTPS>=bin1) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin6) AND NOT(HTTPS>=bin9) AND NOT(DNS>=bin2) AND NOT(DNS>=bin5) AND NOT(DNS>=bin6) AND NOT(DNS>=bin7) AND NOT(SSH>=bin2) AND NOT(SSH>=bin3) AND NOT(DHCP>=bin2) AND NOT(DHCP>=bin5) AND NOT(DHCP>=bin7) AND NOT(DHCP>=bin8) AND NOT(ARP>=bin3) AND NOT(ARP>=bin4) AND NOT(ARP>=bin5) AND NOT(ARP>=bin6) AND NOT(ICMP>=bin2) AND NOT(ICMP>=bin3) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin5) AND NOT(ICMP>=bin8) AND NOT(IGMP>=bin6) AND NOT(IGMP>=bin8)

- clause #548 (pi_hat=1.0, coef=+0.3158, positive, 14 literals):
  IF Rate>=bin4 AND syn_flag_number>=bin1 AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin7) AND NOT(Rate>=bin8) AND NOT(syn_flag_number>=bin5) AND NOT(ack_flag_number>=bin1) AND NOT(rst_count>=bin1) AND NOT(HTTP>=bin8) AND NOT(HTTPS>=bin4) AND NOT(DNS>=bin9) AND NOT(SSH>=bin5) AND NOT(IGMP>=bin5) AND NOT(IGMP>=bin6)

- clause #1042 (pi_hat=1.0, coef=+0.6588, negative, 93 literals)  [from DoS's bank]:
  IF Protocol Type>=bin1 AND Rate>=bin6 AND Rate>=bin7 AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin5) AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin9) AND NOT(Protocol Type>=bin3) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin3) AND NOT(fin_flag_number>=bin5) AND NOT(fin_flag_number>=bin7) AND NOT(fin_flag_number>=bin9) AND NOT(syn_flag_number>=bin3) AND NOT(syn_flag_number>=bin4) AND NOT(syn_flag_number>=bin5) AND NOT(syn_flag_number>=bin6) AND NOT(syn_flag_number>=bin7) AND NOT(syn_flag_number>=bin8) AND NOT(rst_flag_number>=bin3) AND NOT(rst_flag_number>=bin4) AND NOT(rst_flag_number>=bin6) AND NOT(rst_flag_number>=bin7) AND NOT(rst_flag_number>=bin8) AND NOT(rst_flag_number>=bin9) AND NOT(psh_flag_number>=bin2) AND NOT(psh_flag_number>=bin3) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin5) AND NOT(psh_flag_number>=bin6) AND NOT(psh_flag_number>=bin7) AND NOT(psh_flag_number>=bin8) AND NOT(psh_flag_number>=bin9) AND NOT(ack_flag_number>=bin1) AND NOT(ack_flag_number>=bin3) AND NOT(ack_flag_number>=bin4) AND NOT(ack_flag_number>=bin5) AND NOT(ack_flag_number>=bin6) AND NOT(ack_flag_number>=bin7) AND NOT(ack_flag_number>=bin8) AND NOT(ece_flag_number>=bin2) AND NOT(ece_flag_number>=bin3) AND NOT(cwr_flag_number>=bin1) AND NOT(rst_count>=bin4) AND NOT(rst_count>=bin5) AND NOT(rst_count>=bin6) AND NOT(rst_count>=bin8) AND NOT(rst_count>=bin9) AND NOT(HTTP>=bin2) AND NOT(HTTP>=bin3) AND NOT(HTTP>=bin4) AND NOT(HTTP>=bin5) AND NOT(HTTP>=bin7) AND NOT(HTTP>=bin8) AND NOT(HTTP>=bin9) AND NOT(HTTPS>=bin1) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin4) AND NOT(HTTPS>=bin8) AND NOT(DNS>=bin1) AND NOT(DNS>=bin2) AND NOT(DNS>=bin3) AND NOT(DNS>=bin5) AND NOT(DNS>=bin9) AND NOT(SMTP>=bin1) AND NOT(SSH>=bin5) AND NOT(IRC>=bin1) AND NOT(IRC>=bin2) AND NOT(DHCP>=bin1) AND NOT(DHCP>=bin2) AND NOT(DHCP>=bin3) AND NOT(DHCP>=bin4) AND NOT(DHCP>=bin6) AND NOT(DHCP>=bin7) AND NOT(DHCP>=bin8) AND NOT(ARP>=bin1) AND NOT(ARP>=bin3) AND NOT(ARP>=bin4) AND NOT(ARP>=bin5) AND NOT(ARP>=bin6) AND NOT(ARP>=bin7) AND NOT(ARP>=bin8) AND NOT(ICMP>=bin1) AND NOT(ICMP>=bin2) AND NOT(ICMP>=bin5) AND NOT(ICMP>=bin6) AND NOT(ICMP>=bin8) AND NOT(IGMP>=bin3) AND NOT(IGMP>=bin4) AND NOT(IGMP>=bin5) AND NOT(IGMP>=bin6) AND NOT(IGMP>=bin8) AND NOT(IGMP>=bin9)

- clause #1197 (pi_hat=1.0, coef=+0.3589, negative, 91 literals)  [from DoS's bank]:
  IF Rate>=bin2 AND Rate>=bin4 AND Rate>=bin5 AND Rate>=bin7 AND Rate>=bin8 AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin9) AND NOT(Protocol Type>=bin3) AND NOT(fin_flag_number>=bin2) AND NOT(fin_flag_number>=bin3) AND NOT(fin_flag_number>=bin6) AND NOT(fin_flag_number>=bin7) AND NOT(fin_flag_number>=bin9) AND NOT(syn_flag_number>=bin3) AND NOT(syn_flag_number>=bin4) AND NOT(syn_flag_number>=bin5) AND NOT(syn_flag_number>=bin7) AND NOT(syn_flag_number>=bin8) AND NOT(syn_flag_number>=bin9) AND NOT(rst_flag_number>=bin3) AND NOT(rst_flag_number>=bin4) AND NOT(rst_flag_number>=bin6) AND NOT(rst_flag_number>=bin7) AND NOT(rst_flag_number>=bin8) AND NOT(rst_flag_number>=bin9) AND NOT(psh_flag_number>=bin1) AND NOT(psh_flag_number>=bin2) AND NOT(psh_flag_number>=bin3) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin5) AND NOT(psh_flag_number>=bin7) AND NOT(psh_flag_number>=bin9) AND NOT(ack_flag_number>=bin5) AND NOT(ack_flag_number>=bin6) AND NOT(ack_flag_number>=bin7) AND NOT(ack_flag_number>=bin8) AND NOT(ack_flag_number>=bin9) AND NOT(ece_flag_number>=bin1) AND NOT(ece_flag_number>=bin2) AND NOT(ece_flag_number>=bin3) AND NOT(cwr_flag_number>=bin2) AND NOT(rst_count>=bin4) AND NOT(rst_count>=bin5) AND NOT(rst_count>=bin7) AND NOT(HTTP>=bin1) AND NOT(HTTP>=bin2) AND NOT(HTTP>=bin3) AND NOT(HTTP>=bin5) AND NOT(HTTP>=bin6) AND NOT(HTTP>=bin7) AND NOT(HTTP>=bin8) AND NOT(HTTPS>=bin2) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin4) AND NOT(HTTPS>=bin5) AND NOT(HTTPS>=bin6) AND NOT(HTTPS>=bin7) AND NOT(HTTPS>=bin8) AND NOT(HTTPS>=bin9) AND NOT(DNS>=bin2) AND NOT(DNS>=bin6) AND NOT(DNS>=bin8) AND NOT(Telnet>=bin2) AND NOT(SMTP>=bin1) AND NOT(SMTP>=bin2) AND NOT(SSH>=bin1) AND NOT(SSH>=bin3) AND NOT(SSH>=bin4) AND NOT(IRC>=bin1) AND NOT(IRC>=bin2) AND NOT(DHCP>=bin3) AND NOT(DHCP>=bin6) AND NOT(DHCP>=bin7) AND NOT(DHCP>=bin9) AND NOT(ARP>=bin1) AND NOT(ARP>=bin2) AND NOT(ARP>=bin4) AND NOT(ARP>=bin5) AND NOT(ARP>=bin6) AND NOT(ARP>=bin8) AND NOT(ARP>=bin9) AND NOT(ICMP>=bin3) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin5) AND NOT(ICMP>=bin6) AND NOT(ICMP>=bin7) AND NOT(ICMP>=bin9) AND NOT(IGMP>=bin2) AND NOT(IGMP>=bin4) AND NOT(IGMP>=bin5) AND NOT(IGMP>=bin6)

- clause #1626 (pi_hat=1.0, coef=+0.0463, positive, 1 literals)  [from Recon's bank]:
  IF Rate>=bin6

- clause #1765 (pi_hat=1.0, coef=+0.0546, positive, 1 literals)  [from Recon's bank]:
  IF Rate>=bin6

  ... and 14 more (see JSON for full list)


## Family: DoS  (33 inculpatory + 0 exculpatory signature clauses)

(5/33 clauses drawn from DoS's own clause bank; 28 drawn from other classes' banks)

**Inculpatory (evidence FOR this family):**

- clause #332 (pi_hat=1.0, coef=+0.1473, negative, 4 literals)  [from Benign's bank]:
  IF Rate>=bin3 AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin7)

- clause #354 (pi_hat=1.0, coef=+0.5558, negative, 8 literals)  [from Benign's bank]:
  IF Rate>=bin3 AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin7) AND NOT(psh_flag_number>=bin1) AND NOT(psh_flag_number>=bin2) AND NOT(ack_flag_number>=bin2) AND NOT(ack_flag_number>=bin3)

- clause #386 (pi_hat=1.0, coef=+0.2129, negative, 2 literals)  [from Benign's bank]:
  IF Rate>=bin2 AND NOT(Header_Length>=bin5)

- clause #455 (pi_hat=1.0, coef=+0.2133, positive, 3 literals)  [from DDoS's bank]:
  IF syn_flag_number>=bin1 AND ack_flag_number>=bin1 AND NOT(Header_Length>=bin7)

- clause #499 (pi_hat=1.0, coef=+0.3533, positive, 2 literals)  [from DDoS's bank]:
  IF rst_count>=bin2 AND NOT(ack_flag_number>=bin3)

- clause #533 (pi_hat=1.0, coef=+0.2001, positive, 8 literals)  [from DDoS's bank]:
  IF syn_flag_number>=bin2 AND ack_flag_number>=bin1 AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin7) AND NOT(Header_Length>=bin8) AND NOT(psh_flag_number>=bin6) AND NOT(HTTPS>=bin6) AND NOT(DNS>=bin7)

- clause #662 (pi_hat=1.0, coef=+0.2051, negative, 36 literals)  [from DDoS's bank]:
  IF Rate>=bin2 AND Rate>=bin4 AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin7) AND NOT(Protocol Type>=bin1) AND NOT(Rate>=bin5) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin5) AND NOT(syn_flag_number>=bin4) AND NOT(syn_flag_number>=bin8) AND NOT(rst_flag_number>=bin2) AND NOT(psh_flag_number>=bin2) AND NOT(psh_flag_number>=bin6) AND NOT(ack_flag_number>=bin4) AND NOT(ack_flag_number>=bin5) AND NOT(ack_flag_number>=bin7) AND NOT(ack_flag_number>=bin8) AND NOT(ece_flag_number>=bin1) AND NOT(cwr_flag_number>=bin1) AND NOT(HTTP>=bin1) AND NOT(DNS>=bin1) AND NOT(DNS>=bin4) AND NOT(DNS>=bin5) AND NOT(DNS>=bin6) AND NOT(SSH>=bin4) AND NOT(IRC>=bin2) AND NOT(DHCP>=bin6) AND NOT(DHCP>=bin8) AND NOT(ARP>=bin4) AND NOT(ARP>=bin6) AND NOT(ICMP>=bin2) AND NOT(ICMP>=bin6) AND NOT(IGMP>=bin2) AND NOT(IGMP>=bin4) AND NOT(IGMP>=bin5)

- clause #804 (pi_hat=1.0, coef=+0.4317, positive, 3 literals):
  IF syn_flag_number>=bin1 AND ack_flag_number>=bin1 AND NOT(Header_Length>=bin5)

- clause #891 (pi_hat=1.0, coef=+0.4340, positive, 87 literals):
  IF Protocol Type>=bin1 AND Rate>=bin1 AND Rate>=bin2 AND Rate>=bin3 AND NOT(Header_Length>=bin2) AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin8) AND NOT(Header_Length>=bin9) AND NOT(Protocol Type>=bin2) AND NOT(Rate>=bin7) AND NOT(Rate>=bin9) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin3) AND NOT(fin_flag_number>=bin4) AND NOT(fin_flag_number>=bin7) AND NOT(fin_flag_number>=bin9) AND NOT(syn_flag_number>=bin2) AND NOT(syn_flag_number>=bin3) AND NOT(syn_flag_number>=bin6) AND NOT(syn_flag_number>=bin7) AND NOT(syn_flag_number>=bin8) AND NOT(syn_flag_number>=bin9) AND NOT(rst_flag_number>=bin1) AND NOT(rst_flag_number>=bin3) AND NOT(rst_flag_number>=bin4) AND NOT(rst_flag_number>=bin6) AND NOT(rst_flag_number>=bin7) AND NOT(psh_flag_number>=bin1) AND NOT(psh_flag_number>=bin2) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin6) AND NOT(psh_flag_number>=bin7) AND NOT(psh_flag_number>=bin8) AND NOT(ack_flag_number>=bin5) AND NOT(ack_flag_number>=bin8) AND NOT(ack_flag_number>=bin9) AND NOT(ece_flag_number>=bin3) AND NOT(cwr_flag_number>=bin1) AND NOT(rst_count>=bin1) AND NOT(rst_count>=bin2) AND NOT(rst_count>=bin3) AND NOT(rst_count>=bin4) AND NOT(rst_count>=bin7) AND NOT(HTTP>=bin1) AND NOT(HTTP>=bin2) AND NOT(HTTP>=bin3) AND NOT(HTTP>=bin5) AND NOT(HTTP>=bin6) AND NOT(HTTP>=bin9) AND NOT(HTTPS>=bin1) AND NOT(HTTPS>=bin2) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin4) AND NOT(HTTPS>=bin8) AND NOT(DNS>=bin1) AND NOT(DNS>=bin2) AND NOT(DNS>=bin6) AND NOT(DNS>=bin7) AND NOT(DNS>=bin8) AND NOT(DNS>=bin9) AND NOT(Telnet>=bin2) AND NOT(SMTP>=bin1) AND NOT(SMTP>=bin2) AND NOT(SSH>=bin1) AND NOT(SSH>=bin2) AND NOT(SSH>=bin4) AND NOT(IRC>=bin1) AND NOT(DHCP>=bin2) AND NOT(DHCP>=bin4) AND NOT(DHCP>=bin6) AND NOT(DHCP>=bin9) AND NOT(ARP>=bin1) AND NOT(ARP>=bin2) AND NOT(ARP>=bin4) AND NOT(ARP>=bin5) AND NOT(ARP>=bin7) AND NOT(ARP>=bin8) AND NOT(ICMP>=bin1) AND NOT(ICMP>=bin2) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin5) AND NOT(ICMP>=bin6) AND NOT(IGMP>=bin1) AND NOT(IGMP>=bin2) AND NOT(IGMP>=bin3) AND NOT(IGMP>=bin7)

- clause #892 (pi_hat=1.0, coef=+0.1464, positive, 21 literals):
  IF Header_Length>=bin1 AND Header_Length>=bin3 AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin5) AND NOT(Rate>=bin7) AND NOT(fin_flag_number>=bin1) AND NOT(syn_flag_number>=bin1) AND NOT(syn_flag_number>=bin2) AND NOT(rst_flag_number>=bin1) AND NOT(rst_flag_number>=bin8) AND NOT(ack_flag_number>=bin1) AND NOT(ack_flag_number>=bin6) AND NOT(ece_flag_number>=bin1) AND NOT(HTTP>=bin1) AND NOT(HTTP>=bin2) AND NOT(DNS>=bin2) AND NOT(SMTP>=bin2) AND NOT(SSH>=bin1) AND NOT(DHCP>=bin8) AND NOT(ICMP>=bin7) AND NOT(IGMP>=bin5)

- clause #1412 (pi_hat=1.0, coef=+0.1191, negative, 3 literals)  [from MQTT's bank]:
  IF Rate>=bin2 AND NOT(Header_Length>=bin6) AND NOT(psh_flag_number>=bin3)

- clause #1837 (pi_hat=1.0, coef=+0.1910, negative, 26 literals)  [from Recon's bank]:
  IF Header_Length>=bin2 AND Rate>=bin1 AND NOT(Header_Length>=bin4) AND NOT(Rate>=bin6) AND NOT(syn_flag_number>=bin4) AND NOT(rst_flag_number>=bin1) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin9) AND NOT(ack_flag_number>=bin4) AND NOT(ack_flag_number>=bin7) AND NOT(rst_count>=bin1) AND NOT(rst_count>=bin4) AND NOT(HTTP>=bin1) AND NOT(HTTP>=bin5) AND NOT(HTTPS>=bin9) AND NOT(DNS>=bin2) AND NOT(DNS>=bin5) AND NOT(DNS>=bin8) AND NOT(Telnet>=bin2) AND NOT(DHCP>=bin3) AND NOT(DHCP>=bin6) AND NOT(DHCP>=bin9) AND NOT(ARP>=bin7) AND NOT(ARP>=bin9) AND NOT(IGMP>=bin3) AND NOT(IGMP>=bin4)

- clause #1948 (pi_hat=1.0, coef=+0.1323, negative, 58 literals)  [from Recon's bank]:
  IF Rate>=bin2 AND NOT(Header_Length>=bin4) AND NOT(Header_Length>=bin8) AND NOT(Protocol Type>=bin3) AND NOT(Rate>=bin4) AND NOT(Rate>=bin6) AND NOT(Rate>=bin8) AND NOT(Rate>=bin9) AND NOT(fin_flag_number>=bin6) AND NOT(syn_flag_number>=bin4) AND NOT(syn_flag_number>=bin6) AND NOT(syn_flag_number>=bin8) AND NOT(syn_flag_number>=bin9) AND NOT(rst_flag_number>=bin2) AND NOT(rst_flag_number>=bin3) AND NOT(rst_flag_number>=bin6) AND NOT(rst_flag_number>=bin7) AND NOT(rst_flag_number>=bin8) AND NOT(psh_flag_number>=bin1) AND NOT(psh_flag_number>=bin2) AND NOT(psh_flag_number>=bin5) AND NOT(psh_flag_number>=bin8) AND NOT(psh_flag_number>=bin9) AND NOT(ack_flag_number>=bin1) AND NOT(ack_flag_number>=bin6) AND NOT(rst_count>=bin1) AND NOT(rst_count>=bin4) AND NOT(rst_count>=bin7) AND NOT(HTTP>=bin2) AND NOT(HTTP>=bin4) AND NOT(HTTP>=bin8) AND NOT(HTTPS>=bin2) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin6) AND NOT(DNS>=bin1) AND NOT(DNS>=bin2) AND NOT(DNS>=bin5) AND NOT(DNS>=bin8) AND NOT(Telnet>=bin1) AND NOT(Telnet>=bin2) AND NOT(SMTP>=bin1) AND NOT(SSH>=bin1) AND NOT(SSH>=bin4) AND NOT(SSH>=bin5) AND NOT(IRC>=bin1) AND NOT(DHCP>=bin2) AND NOT(DHCP>=bin3) AND NOT(ARP>=bin1) AND NOT(ARP>=bin5) AND NOT(ARP>=bin8) AND NOT(ICMP>=bin2) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin6) AND NOT(ICMP>=bin9) AND NOT(IGMP>=bin1) AND NOT(IGMP>=bin5) AND NOT(IGMP>=bin7) AND NOT(IGMP>=bin8)

- clause #1999 (pi_hat=1.0, coef=+0.3781, negative, 12 literals)  [from Recon's bank]:
  IF Rate>=bin1 AND Rate>=bin2 AND NOT(Protocol Type>=bin3) AND NOT(syn_flag_number>=bin1) AND NOT(ack_flag_number>=bin3) AND NOT(ece_flag_number>=bin2) AND NOT(rst_count>=bin8) AND NOT(HTTPS>=bin8) AND NOT(DNS>=bin9) AND NOT(ARP>=bin9) AND NOT(IGMP>=bin6) AND NOT(IGMP>=bin7)

- clause #2081 (pi_hat=1.0, coef=+0.3832, positive, 12 literals)  [from Spoofing's bank]:
  IF Header_Length>=bin1 AND Rate>=bin2 AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin7) AND NOT(Rate>=bin6) AND NOT(syn_flag_number>=bin1) AND NOT(rst_flag_number>=bin1) AND NOT(DNS>=bin2) AND NOT(DNS>=bin7) AND NOT(SSH>=bin1) AND NOT(ICMP>=bin1) AND NOT(IGMP>=bin4)

  ... and 18 more (see JSON for full list)


## Family: MQTT  (53 inculpatory + 0 exculpatory signature clauses)

(14/53 clauses drawn from MQTT's own clause bank; 39 drawn from other classes' banks)

**Inculpatory (evidence FOR this family):**

- clause #21 (pi_hat=1.0, coef=+0.0975, positive, 7 literals)  [from Benign's bank]:
  IF psh_flag_number>=bin2 AND NOT(Protocol Type>=bin3) AND NOT(syn_flag_number>=bin5) AND NOT(ack_flag_number>=bin6) AND NOT(ece_flag_number>=bin2) AND NOT(HTTPS>=bin1) AND NOT(IRC>=bin1)

- clause #93 (pi_hat=1.0, coef=+0.0261, positive, 1 literals)  [from Benign's bank]:
  IF Rate>=bin9

- clause #495 (pi_hat=1.0, coef=+0.7663, positive, 2 literals)  [from DDoS's bank]:
  IF rst_flag_number>=bin1 AND NOT(ack_flag_number>=bin2)

- clause #916 (pi_hat=1.0, coef=+0.0234, positive, 1 literals)  [from DoS's bank]:
  IF Rate>=bin9

- clause #920 (pi_hat=1.0, coef=+0.0356, positive, 1 literals)  [from DoS's bank]:
  IF Rate>=bin9

- clause #992 (pi_hat=1.0, coef=+0.1627, positive, 4 literals)  [from DoS's bank]:
  IF Rate>=bin2 AND NOT(Rate>=bin3) AND NOT(rst_count>=bin7) AND NOT(DNS>=bin4)

- clause #1204 (pi_hat=1.0, coef=+1.0970, positive, 2 literals):
  IF Rate>=bin2 AND fin_flag_number>=bin1

- clause #1253 (pi_hat=1.0, coef=+1.1494, positive, 3 literals):
  IF rst_count>=bin1 AND NOT(syn_flag_number>=bin2) AND NOT(ack_flag_number>=bin4)

- clause #1272 (pi_hat=1.0, coef=+0.8810, positive, 2 literals):
  IF Header_Length>=bin5 AND syn_flag_number>=bin1

- clause #1286 (pi_hat=1.0, coef=+0.1022, positive, 3 literals):
  IF Header_Length>=bin4 AND Rate>=bin1 AND rst_count>=bin1

- clause #1291 (pi_hat=1.0, coef=+1.1416, positive, 3 literals):
  IF rst_count>=bin1 AND NOT(syn_flag_number>=bin2) AND NOT(ack_flag_number>=bin4)

- clause #1338 (pi_hat=1.0, coef=+0.1451, positive, 3 literals):
  IF Rate>=bin1 AND rst_flag_number>=bin1 AND NOT(ack_flag_number>=bin4)

- clause #1354 (pi_hat=1.0, coef=+0.3879, positive, 7 literals):
  IF Header_Length>=bin2 AND Header_Length>=bin6 AND NOT(psh_flag_number>=bin3) AND NOT(ack_flag_number>=bin5) AND NOT(ack_flag_number>=bin7) AND NOT(HTTPS>=bin1) AND NOT(DHCP>=bin5)

- clause #1356 (pi_hat=1.0, coef=+0.4146, positive, 5 literals):
  IF Header_Length>=bin5 AND Header_Length>=bin6 AND NOT(psh_flag_number>=bin3) AND NOT(HTTP>=bin2) AND NOT(HTTPS>=bin1)

- clause #1360 (pi_hat=1.0, coef=+0.2095, positive, 2 literals):
  IF Header_Length>=bin4 AND Rate>=bin5

  ... and 38 more (see JSON for full list)


## Family: Recon  (28 inculpatory + 0 exculpatory signature clauses)

(14/28 clauses drawn from Recon's own clause bank; 14 drawn from other classes' banks)

**Inculpatory (evidence FOR this family):**

- clause #228 (pi_hat=1.0, coef=+0.3508, negative, 6 literals)  [from Benign's bank]:
  IF Header_Length>=bin2 AND NOT(Header_Length>=bin5) AND NOT(Header_Length>=bin6) AND NOT(syn_flag_number>=bin4) AND NOT(rst_flag_number>=bin9) AND NOT(HTTPS>=bin1)

- clause #1408 (pi_hat=1.0, coef=+0.6994, negative, 2 literals)  [from MQTT's bank]:
  IF syn_flag_number>=bin2 AND NOT(ack_flag_number>=bin2)

- clause #1441 (pi_hat=1.0, coef=+0.4223, negative, 2 literals)  [from MQTT's bank]:
  IF rst_flag_number>=bin2 AND NOT(Header_Length>=bin4)

- clause #1457 (pi_hat=1.0, coef=+0.9716, negative, 3 literals)  [from MQTT's bank]:
  IF Rate>=bin1 AND NOT(Header_Length>=bin6) AND NOT(rst_count>=bin3)

- clause #1472 (pi_hat=1.0, coef=+0.1643, negative, 2 literals)  [from MQTT's bank]:
  IF rst_flag_number>=bin1 AND NOT(Header_Length>=bin4)

- clause #1504 (pi_hat=1.0, coef=+0.1605, negative, 2 literals)  [from MQTT's bank]:
  IF rst_count>=bin1 AND NOT(Header_Length>=bin4)

- clause #1513 (pi_hat=1.0, coef=+0.1675, negative, 2 literals)  [from MQTT's bank]:
  IF rst_count>=bin1 AND NOT(Header_Length>=bin4)

- clause #1521 (pi_hat=1.0, coef=+0.3366, negative, 2 literals)  [from MQTT's bank]:
  IF rst_flag_number>=bin1 AND NOT(Rate>=bin2)

- clause #1650 (pi_hat=1.0, coef=+0.0855, positive, 2 literals):
  IF rst_flag_number>=bin2 AND NOT(rst_count>=bin3)

- clause #1675 (pi_hat=1.0, coef=+0.8141, positive, 16 literals):
  IF Rate>=bin3 AND syn_flag_number>=bin1 AND syn_flag_number>=bin2 AND NOT(Header_Length>=bin5) AND NOT(Header_Length>=bin7) AND NOT(Header_Length>=bin8) AND NOT(fin_flag_number>=bin5) AND NOT(psh_flag_number>=bin6) AND NOT(ack_flag_number>=bin1) AND NOT(ack_flag_number>=bin2) AND NOT(rst_count>=bin2) AND NOT(rst_count>=bin3) AND NOT(HTTP>=bin9) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin6) AND NOT(DNS>=bin5)

- clause #1683 (pi_hat=1.0, coef=+0.6091, positive, 2 literals):
  IF rst_flag_number>=bin3 AND ack_flag_number>=bin2

- clause #1689 (pi_hat=1.0, coef=+0.6538, positive, 2 literals):
  IF rst_flag_number>=bin3 AND ack_flag_number>=bin2

- clause #1709 (pi_hat=1.0, coef=+0.1502, positive, 9 literals):
  IF Header_Length>=bin2 AND Header_Length>=bin3 AND syn_flag_number>=bin2 AND NOT(psh_flag_number>=bin1) AND NOT(ack_flag_number>=bin1) AND NOT(DNS>=bin1) AND NOT(DNS>=bin8) AND NOT(ARP>=bin3) AND NOT(ICMP>=bin6)

- clause #1730 (pi_hat=1.0, coef=+0.4287, positive, 2 literals):
  IF rst_flag_number>=bin2 AND NOT(Header_Length>=bin4)

- clause #1735 (pi_hat=1.0, coef=+0.1388, positive, 109 literals):
  IF Header_Length>=bin2 AND Header_Length>=bin3 AND syn_flag_number>=bin1 AND syn_flag_number>=bin2 AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin8) AND NOT(Header_Length>=bin9) AND NOT(Protocol Type>=bin2) AND NOT(Protocol Type>=bin3) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin2) AND NOT(fin_flag_number>=bin3) AND NOT(fin_flag_number>=bin4) AND NOT(fin_flag_number>=bin5) AND NOT(fin_flag_number>=bin7) AND NOT(fin_flag_number>=bin8) AND NOT(fin_flag_number>=bin9) AND NOT(syn_flag_number>=bin3) AND NOT(syn_flag_number>=bin4) AND NOT(syn_flag_number>=bin5) AND NOT(syn_flag_number>=bin6) AND NOT(syn_flag_number>=bin7) AND NOT(syn_flag_number>=bin9) AND NOT(rst_flag_number>=bin1) AND NOT(rst_flag_number>=bin2) AND NOT(rst_flag_number>=bin4) AND NOT(rst_flag_number>=bin5) AND NOT(rst_flag_number>=bin7) AND NOT(rst_flag_number>=bin8) AND NOT(rst_flag_number>=bin9) AND NOT(psh_flag_number>=bin1) AND NOT(psh_flag_number>=bin2) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin5) AND NOT(psh_flag_number>=bin6) AND NOT(psh_flag_number>=bin7) AND NOT(psh_flag_number>=bin8) AND NOT(psh_flag_number>=bin9) AND NOT(ack_flag_number>=bin1) AND NOT(ack_flag_number>=bin2) AND NOT(ack_flag_number>=bin3) AND NOT(ack_flag_number>=bin4) AND NOT(ack_flag_number>=bin5) AND NOT(ack_flag_number>=bin6) AND NOT(ack_flag_number>=bin7) AND NOT(ack_flag_number>=bin8) AND NOT(ack_flag_number>=bin9) AND NOT(ece_flag_number>=bin1) AND NOT(ece_flag_number>=bin2) AND NOT(ece_flag_number>=bin3) AND NOT(cwr_flag_number>=bin1) AND NOT(cwr_flag_number>=bin2) AND NOT(rst_count>=bin1) AND NOT(rst_count>=bin2) AND NOT(rst_count>=bin3) AND NOT(rst_count>=bin5) AND NOT(rst_count>=bin7) AND NOT(rst_count>=bin8) AND NOT(rst_count>=bin9) AND NOT(HTTP>=bin1) AND NOT(HTTP>=bin2) AND NOT(HTTP>=bin3) AND NOT(HTTP>=bin5) AND NOT(HTTP>=bin6) AND NOT(HTTP>=bin7) AND NOT(HTTP>=bin9) AND NOT(HTTPS>=bin1) AND NOT(HTTPS>=bin2) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin4) AND NOT(HTTPS>=bin6) AND NOT(DNS>=bin1) AND NOT(DNS>=bin2) AND NOT(DNS>=bin3) AND NOT(DNS>=bin5) AND NOT(DNS>=bin6) AND NOT(DNS>=bin9) AND NOT(Telnet>=bin1) AND NOT(SMTP>=bin1) AND NOT(SSH>=bin1) AND NOT(SSH>=bin2) AND NOT(SSH>=bin3) AND NOT(SSH>=bin4) AND NOT(SSH>=bin5) AND NOT(IRC>=bin1) AND NOT(IRC>=bin2) AND NOT(DHCP>=bin1) AND NOT(DHCP>=bin2) AND NOT(DHCP>=bin4) AND NOT(DHCP>=bin6) AND NOT(DHCP>=bin8) AND NOT(DHCP>=bin9) AND NOT(ARP>=bin1) AND NOT(ARP>=bin3) AND NOT(ARP>=bin4) AND NOT(ARP>=bin5) AND NOT(ARP>=bin6) AND NOT(ARP>=bin7) AND NOT(ARP>=bin9) AND NOT(ICMP>=bin2) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin6) AND NOT(ICMP>=bin8) AND NOT(ICMP>=bin9) AND NOT(IGMP>=bin1) AND NOT(IGMP>=bin3) AND NOT(IGMP>=bin4) AND NOT(IGMP>=bin5) AND NOT(IGMP>=bin9)

  ... and 13 more (see JSON for full list)


## Family: Spoofing  (50 inculpatory + 0 exculpatory signature clauses)

(11/50 clauses drawn from Spoofing's own clause bank; 39 drawn from other classes' banks)

**Inculpatory (evidence FOR this family):**

- clause #74 (pi_hat=1.0, coef=+0.2272, positive, 92 literals)  [from Benign's bank]:
  IF Header_Length>=bin1 AND Header_Length>=bin2 AND Header_Length>=bin3 AND Header_Length>=bin4 AND Header_Length>=bin5 AND Header_Length>=bin6 AND Protocol Type>=bin1 AND Rate>=bin1 AND psh_flag_number>=bin1 AND ack_flag_number>=bin1 AND ack_flag_number>=bin2 AND ack_flag_number>=bin3 AND ack_flag_number>=bin4 AND NOT(Header_Length>=bin8) AND NOT(Header_Length>=bin9) AND NOT(Rate>=bin6) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin3) AND NOT(fin_flag_number>=bin4) AND NOT(fin_flag_number>=bin5) AND NOT(fin_flag_number>=bin6) AND NOT(fin_flag_number>=bin7) AND NOT(fin_flag_number>=bin8) AND NOT(fin_flag_number>=bin9) AND NOT(syn_flag_number>=bin1) AND NOT(syn_flag_number>=bin2) AND NOT(syn_flag_number>=bin3) AND NOT(syn_flag_number>=bin4) AND NOT(syn_flag_number>=bin5) AND NOT(syn_flag_number>=bin7) AND NOT(syn_flag_number>=bin9) AND NOT(rst_flag_number>=bin3) AND NOT(rst_flag_number>=bin5) AND NOT(rst_flag_number>=bin7) AND NOT(rst_flag_number>=bin8) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin5) AND NOT(psh_flag_number>=bin7) AND NOT(psh_flag_number>=bin8) AND NOT(psh_flag_number>=bin9) AND NOT(ack_flag_number>=bin7) AND NOT(ece_flag_number>=bin3) AND NOT(rst_count>=bin2) AND NOT(rst_count>=bin3) AND NOT(rst_count>=bin4) AND NOT(rst_count>=bin6) AND NOT(rst_count>=bin7) AND NOT(HTTP>=bin2) AND NOT(HTTP>=bin3) AND NOT(HTTP>=bin4) AND NOT(HTTP>=bin5) AND NOT(HTTP>=bin6) AND NOT(HTTPS>=bin2) AND NOT(HTTPS>=bin3) AND NOT(HTTPS>=bin4) AND NOT(HTTPS>=bin8) AND NOT(HTTPS>=bin9) AND NOT(DNS>=bin1) AND NOT(DNS>=bin2) AND NOT(DNS>=bin4) AND NOT(DNS>=bin8) AND NOT(DNS>=bin9) AND NOT(SMTP>=bin2) AND NOT(SSH>=bin2) AND NOT(SSH>=bin3) AND NOT(SSH>=bin4) AND NOT(SSH>=bin5) AND NOT(IRC>=bin2) AND NOT(DHCP>=bin1) AND NOT(DHCP>=bin4) AND NOT(DHCP>=bin5) AND NOT(DHCP>=bin6) AND NOT(DHCP>=bin7) AND NOT(DHCP>=bin9) AND NOT(ARP>=bin1) AND NOT(ARP>=bin3) AND NOT(ARP>=bin4) AND NOT(ARP>=bin5) AND NOT(ARP>=bin6) AND NOT(ICMP>=bin1) AND NOT(ICMP>=bin2) AND NOT(ICMP>=bin3) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin5) AND NOT(ICMP>=bin6) AND NOT(ICMP>=bin7) AND NOT(IGMP>=bin2) AND NOT(IGMP>=bin3) AND NOT(IGMP>=bin4) AND NOT(IGMP>=bin5) AND NOT(IGMP>=bin8) AND NOT(IGMP>=bin9)

- clause #75 (pi_hat=1.0, coef=+0.2133, positive, 13 literals)  [from Benign's bank]:
  IF Header_Length>=bin2 AND Rate>=bin1 AND psh_flag_number>=bin1 AND NOT(Rate>=bin2) AND NOT(Rate>=bin3) AND NOT(Rate>=bin7) AND NOT(Rate>=bin8) AND NOT(syn_flag_number>=bin2) AND NOT(ack_flag_number>=bin6) AND NOT(HTTPS>=bin5) AND NOT(DHCP>=bin7) AND NOT(ICMP>=bin2) AND NOT(IGMP>=bin4)

- clause #78 (pi_hat=1.0, coef=+0.0713, positive, 1 literals)  [from Benign's bank]:
  IF HTTPS>=bin1

- clause #85 (pi_hat=1.0, coef=+0.1902, positive, 19 literals)  [from Benign's bank]:
  IF Header_Length>=bin1 AND Protocol Type>=bin1 AND NOT(Rate>=bin3) AND NOT(fin_flag_number>=bin5) AND NOT(syn_flag_number>=bin1) AND NOT(rst_flag_number>=bin2) AND NOT(rst_flag_number>=bin3) AND NOT(rst_flag_number>=bin7) AND NOT(rst_flag_number>=bin9) AND NOT(psh_flag_number>=bin3) AND NOT(ack_flag_number>=bin2) AND NOT(ack_flag_number>=bin9) AND NOT(HTTP>=bin5) AND NOT(HTTPS>=bin8) AND NOT(DNS>=bin9) AND NOT(DHCP>=bin4) AND NOT(DHCP>=bin7) AND NOT(ICMP>=bin1) AND NOT(ICMP>=bin5)

- clause #195 (pi_hat=1.0, coef=+0.0657, positive, 1 literals)  [from Benign's bank]:
  IF HTTPS>=bin1

- clause #408 (pi_hat=1.0, coef=+0.0534, positive, 1 literals)  [from DDoS's bank]:
  IF HTTPS>=bin1

- clause #476 (pi_hat=1.0, coef=+0.0442, positive, 1 literals)  [from DDoS's bank]:
  IF HTTPS>=bin1

- clause #485 (pi_hat=1.0, coef=+0.0549, positive, 1 literals)  [from DDoS's bank]:
  IF HTTPS>=bin1

- clause #541 (pi_hat=1.0, coef=+0.0630, positive, 1 literals)  [from DDoS's bank]:
  IF HTTPS>=bin1

- clause #560 (pi_hat=1.0, coef=+0.0680, positive, 1 literals)  [from DDoS's bank]:
  IF HTTPS>=bin1

- clause #566 (pi_hat=1.0, coef=+0.0496, positive, 1 literals)  [from DDoS's bank]:
  IF HTTPS>=bin1

- clause #711 (pi_hat=1.0, coef=+0.7572, negative, 100 literals)  [from DDoS's bank]:
  IF Protocol Type>=bin1 AND Rate>=bin1 AND NOT(Header_Length>=bin2) AND NOT(Header_Length>=bin5) AND NOT(Header_Length>=bin6) AND NOT(Header_Length>=bin7) AND NOT(Header_Length>=bin8) AND NOT(Header_Length>=bin9) AND NOT(Protocol Type>=bin2) AND NOT(Rate>=bin5) AND NOT(Rate>=bin6) AND NOT(fin_flag_number>=bin1) AND NOT(fin_flag_number>=bin2) AND NOT(fin_flag_number>=bin3) AND NOT(fin_flag_number>=bin4) AND NOT(fin_flag_number>=bin5) AND NOT(fin_flag_number>=bin7) AND NOT(fin_flag_number>=bin8) AND NOT(syn_flag_number>=bin1) AND NOT(syn_flag_number>=bin7) AND NOT(rst_flag_number>=bin1) AND NOT(rst_flag_number>=bin2) AND NOT(rst_flag_number>=bin3) AND NOT(rst_flag_number>=bin5) AND NOT(rst_flag_number>=bin8) AND NOT(rst_flag_number>=bin9) AND NOT(psh_flag_number>=bin2) AND NOT(psh_flag_number>=bin3) AND NOT(psh_flag_number>=bin4) AND NOT(psh_flag_number>=bin5) AND NOT(psh_flag_number>=bin6) AND NOT(psh_flag_number>=bin8) AND NOT(ack_flag_number>=bin1) AND NOT(ack_flag_number>=bin2) AND NOT(ack_flag_number>=bin3) AND NOT(ack_flag_number>=bin4) AND NOT(ack_flag_number>=bin6) AND NOT(ack_flag_number>=bin7) AND NOT(ack_flag_number>=bin8) AND NOT(ack_flag_number>=bin9) AND NOT(ece_flag_number>=bin3) AND NOT(cwr_flag_number>=bin1) AND NOT(rst_count>=bin1) AND NOT(rst_count>=bin2) AND NOT(rst_count>=bin3) AND NOT(rst_count>=bin4) AND NOT(rst_count>=bin5) AND NOT(rst_count>=bin6) AND NOT(rst_count>=bin7) AND NOT(rst_count>=bin8) AND NOT(rst_count>=bin9) AND NOT(HTTP>=bin1) AND NOT(HTTP>=bin5) AND NOT(HTTP>=bin6) AND NOT(HTTP>=bin7) AND NOT(HTTP>=bin8) AND NOT(HTTP>=bin9) AND NOT(HTTPS>=bin2) AND NOT(HTTPS>=bin4) AND NOT(HTTPS>=bin5) AND NOT(HTTPS>=bin6) AND NOT(HTTPS>=bin7) AND NOT(HTTPS>=bin8) AND NOT(HTTPS>=bin9) AND NOT(DNS>=bin1) AND NOT(DNS>=bin2) AND NOT(DNS>=bin3) AND NOT(DNS>=bin4) AND NOT(DNS>=bin5) AND NOT(DNS>=bin8) AND NOT(Telnet>=bin1) AND NOT(Telnet>=bin2) AND NOT(SSH>=bin2) AND NOT(SSH>=bin4) AND NOT(SSH>=bin5) AND NOT(IRC>=bin1) AND NOT(IRC>=bin2) AND NOT(DHCP>=bin1) AND NOT(DHCP>=bin2) AND NOT(DHCP>=bin3) AND NOT(DHCP>=bin4) AND NOT(DHCP>=bin9) AND NOT(ARP>=bin1) AND NOT(ARP>=bin2) AND NOT(ARP>=bin3) AND NOT(ARP>=bin4) AND NOT(ARP>=bin5) AND NOT(ARP>=bin6) AND NOT(ARP>=bin7) AND NOT(ARP>=bin8) AND NOT(ICMP>=bin2) AND NOT(ICMP>=bin3) AND NOT(ICMP>=bin4) AND NOT(ICMP>=bin5) AND NOT(ICMP>=bin6) AND NOT(ICMP>=bin7) AND NOT(IGMP>=bin1) AND NOT(IGMP>=bin2) AND NOT(IGMP>=bin3) AND NOT(IGMP>=bin8)

- clause #740 (pi_hat=1.0, coef=+0.4619, negative, 3 literals)  [from DDoS's bank]:
  IF NOT(Header_Length>=bin5) AND NOT(Rate>=bin4) AND NOT(rst_flag_number>=bin3)

- clause #752 (pi_hat=1.0, coef=+0.0809, negative, 3 literals)  [from DDoS's bank]:
  IF NOT(Header_Length>=bin4) AND NOT(Rate>=bin4) AND NOT(ICMP>=bin1)

- clause #806 (pi_hat=1.0, coef=+0.0771, positive, 1 literals)  [from DoS's bank]:
  IF HTTPS>=bin1

  ... and 35 more (see JSON for full list)
